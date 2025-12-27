use ndarray::prelude::*;
use rayon::prelude::*;
use nifti::NiftiHeader;

pub fn postprocess(
    prediction: Array5<f32>, 
    resampled_shape: [usize; 3], 
    ras_original_shape: [usize; 3],
    original_shape: [usize; 3],
    pad_size: [usize; 3],
    header: &NiftiHeader
) -> Array3<u8> {
    // 1. Argmax and Crop (Still in ZYX padded space)
    let c = prediction.shape()[1];
    let z_dim = resampled_shape[2];
    let y_dim = resampled_shape[1];
    let x_dim = resampled_shape[0];

    let cropped_prediction = prediction.slice(s![0, .., ..z_dim, ..y_dim, ..x_dim]);

    // Perform argmax
    let mut seg_resampled_zyx = Array3::<u8>::zeros([z_dim, y_dim, x_dim]);
    seg_resampled_zyx.indexed_iter_mut().par_bridge().for_each(|((z, y, x), val)| {
        let mut max_val = cropped_prediction[[0, z, y, x]];
        let mut max_idx = 0;
        for i in 1..c {
            let v = cropped_prediction[[i, z, y, x]];
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        *val = max_idx as u8;
    });

    // 2. Transpose to (X, Y, Z) for RAS consistency
    let seg_resampled_ras = seg_resampled_zyx.permuted_axes([2, 1, 0]).to_owned();

    // 3. Inverse Resampling to original dimensions IN RAS SPACE
    // This matches Python's change_spacing(..., order=0) on the canonical image
    let mut seg_canonical = Array3::<u8>::zeros(ras_original_shape);

    let scale_x = seg_resampled_ras.shape()[0] as f32 / ras_original_shape[0] as f32;
    let scale_y = seg_resampled_ras.shape()[1] as f32 / ras_original_shape[1] as f32;
    let scale_z = seg_resampled_ras.shape()[2] as f32 / ras_original_shape[2] as f32;

    seg_canonical.indexed_iter_mut().par_bridge().for_each(|((x, y, z), val)| {
        let sx = ((x as f32 * scale_x + 0.5).floor() as usize).min(seg_resampled_ras.shape()[0] - 1);
        let sy = ((y as f32 * scale_y + 0.5).floor() as usize).min(seg_resampled_ras.shape()[1] - 1);
        let sz = ((z as f32 * scale_z + 0.5).floor() as usize).min(seg_resampled_ras.shape()[2] - 1);

        *val = seg_resampled_ras[[sx, sy, sz]];
    });

    // 4. Reorient back from RAS to original
    let final_mask = reorient_back(seg_canonical, header);

    final_mask
}

fn get_orientation_info(header: &NiftiHeader) -> ([usize; 3], [i8; 3]) {
    let mut affine = [[0.0f32; 3]; 3];
    if header.sform_code > 0 {
        affine[0] = [header.srow_x[0], header.srow_x[1], header.srow_x[2]];
        affine[1] = [header.srow_y[0], header.srow_y[1], header.srow_y[2]];
        affine[2] = [header.srow_z[0], header.srow_z[1], header.srow_z[2]];
    } else {
        affine[0][0] = header.pixdim[1];
        affine[1][1] = header.pixdim[2];
        affine[2][2] = header.pixdim[3];
    }

    let mut codes = [0; 3];
    let mut signs = [1; 3];
    for j in 0..3 {
        let mut max_val = 0.0;
        let mut max_idx = 0;
        for i in 0..3 {
            if affine[i][j].abs() > max_val {
                max_val = affine[i][j].abs();
                max_idx = i;
            }
        }
        codes[j] = max_idx;
        signs[j] = if affine[max_idx][j] > 0.0 { 1 } else { -1 };
    }
    (codes, signs)
}

fn reorient_back(mut data: Array3<u8>, header: &NiftiHeader) -> Array3<u8> {
    let (codes, signs) = get_orientation_info(header);

    // 1. Undo permutation
    // `codes` maps original axis index to RAS axis index.
    // e.g., if codes = [2, 1, 0], it means original axis 0 -> RAS axis 2, original axis 1 -> RAS axis 1, original axis 2 -> RAS axis 0.
    // The `data` array is currently in RAS orientation (X, Y, Z).
    // We need to permute it back to the original axis order.
    // If `reorient_to_ras` did `data.permuted_axes(perm)` where `perm[original_axis] = ras_axis`,
    // then to undo, we need `data.permuted_axes(inv_perm)` where `inv_perm[ras_axis] = original_axis`.
    // `codes[original_axis]` gives `ras_axis`.
    // So, `inv_perm[codes[j]] = j` for `j` in `0..3`.
    let mut inv_perm = [0, 0, 0]; // Initialize with dummy values
    for j in 0..3 { // j is the original axis index
        inv_perm[codes[j]] = j; // codes[j] is the RAS axis index
    }
    let mut data = data.permuted_axes(inv_perm);

    // 2. Undo flips
    // `signs` indicates if an axis was flipped in the original data relative to RAS.
    // If `signs[j]` is -1, it means the j-th original axis was flipped.
    // When reorienting to RAS, if `signs[j]` was -1, the `j`-th axis of the *original* data was inverted.
    // Now, `data` is in the original axis order. We need to apply the inverse of the flips.
    // If `signs[j]` is -1, it means the `j`-th axis (in original orientation) needs to be flipped.
    for j in 0..3 {
        if signs[j] == -1 { // If the original j-th axis was flipped relative to RAS
            data.invert_axis(Axis(j));
        }
    }

    data
}
