use ndarray::prelude::*;
use rayon::prelude::*;

use nifti::NiftiHeader;

pub struct PreprocessData {
    pub input_tensor: Array5<f32>,
    pub original_shape: [usize; 3], // The NIfTI shape (un-reoriented)
    pub ras_original_shape: [usize; 3], // The shape after reorientation to RAS
    pub resampled_shape: [usize; 3],
    pub pad_size: [usize; 3],
}

pub fn preprocess(
    mut data: Array3<f32>,
    spacing: [f32; 3],
    header: &NiftiHeader
) -> PreprocessData {
    let input_original_shape = [data.shape()[0], data.shape()[1], data.shape()[2]];

    // 1. Reorient to RAS if necessary
    let (codes, signs) = get_orientation_info(header);
    data = reorient_to_ras_with_info(data, codes, signs);
    
    // Reorient spacing too
    let mut ras_spacing = [0.0f32; 3];
    for j in 0..3 {
        ras_spacing[codes[j]] = spacing[j];
    }
    
    let ras_original_shape = [data.shape()[0], data.shape()[1], data.shape()[2]];
    println!("  RAS Original Shape: {:?}", ras_original_shape);
    println!("  RAS Spacing: {:?}", ras_spacing);

    // 2. Resampling to 3.0mm
    let target_spacing = [3.0, 3.0, 3.0];
    let new_shape = [
        (ras_original_shape[0] as f32 * (ras_spacing[0] / target_spacing[0])).round() as usize,
        (ras_original_shape[1] as f32 * (ras_spacing[1] / target_spacing[1])).round() as usize,
        (ras_original_shape[2] as f32 * (ras_spacing[2] / target_spacing[2])).round() as usize,
    ];

    let scale_factors = [
        new_shape[0] as f32 / ras_original_shape[0] as f32,
        new_shape[1] as f32 / ras_original_shape[1] as f32,
        new_shape[2] as f32 / ras_original_shape[2] as f32,
    ];
    
    let mut resampled = Array3::<f32>::zeros(new_shape);
    
    resampled.indexed_iter_mut().par_bridge().for_each(|((x, y, z), val)| {
        let src_x = (x as f32 + 0.5) / scale_factors[0] - 0.5;
        let src_y = (y as f32 + 0.5) / scale_factors[1] - 0.5;
        let src_z = (z as f32 + 0.5) / scale_factors[2] - 0.5;
        
        let x0_i = src_x.floor() as i32;
        let y0_i = src_y.floor() as i32;
        let z0_i = src_z.floor() as i32;
        
        let dx = src_x - x0_i as f32;
        let dy = src_y - y0_i as f32;
        let dz = src_z - z0_i as f32;

        let mut sample = 0.0;
        
        for i_offset in -1..=2 {
            let xi = (x0_i + i_offset).clamp(0, ras_original_shape[0] as i32 - 1) as usize;
            let wx = cubic_weight(i_offset as f32 - dx);
            
            for j_offset in -1..=2 {
                let yi = (y0_i + j_offset).clamp(0, ras_original_shape[1] as i32 - 1) as usize;
                let wy = cubic_weight(j_offset as f32 - dy);
                
                for k_offset in -1..=2 {
                    let zi = (z0_i + k_offset).clamp(0, ras_original_shape[2] as i32 - 1) as usize;
                    let wz = cubic_weight(k_offset as f32 - dz);
                    
                    sample += data[[xi, yi, zi]] * wx * wy * wz;
                }
            }
        }
        
        *val = sample;
    });

fn cubic_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 {
        (3.0 * x.powi(3) - 6.0 * x.powi(2) + 4.0) / 6.0
    } else if x < 2.0 {
        (2.0 - x).powi(3) / 6.0
    } else {
        0.0
    }
}

    let resampled_shape_ras = [resampled.shape()[0], resampled.shape()[1], resampled.shape()[2]];

    // 3. Normalization (Clip -1004 to 1588, then Z-score)
    resampled.mapv_inplace(|v| {
        let clipped = v.clamp(-1004.0, 1588.0);
        (clipped - (-50.3869)) / 503.3923
    });

    // 4. Transform to ONNX shape (1, 1, D, H, W) and Permute to (Z, Y, X)
    let mut data_reordered = resampled.permuted_axes([2, 1, 0]); // (Z, Y, X)
    
    // 5. Padding to multiple of 32
    let pad_d = (32 - (data_reordered.shape()[0] % 32)) % 32;
    let pad_h = (32 - (data_reordered.shape()[1] % 32)) % 32;
    let pad_w = (32 - (data_reordered.shape()[2] % 32)) % 32;
    
    let padded_shape = [
        data_reordered.shape()[0] + pad_d,
        data_reordered.shape()[1] + pad_h,
        data_reordered.shape()[2] + pad_w,
    ];
    
    let pad_val = *data_reordered.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
    let mut padded = Array3::<f32>::from_elem(padded_shape, pad_val);
    padded.slice_mut(s![..data_reordered.shape()[0], ..data_reordered.shape()[1], ..data_reordered.shape()[2]])
        .assign(&data_reordered);
    
    let input_tensor = padded.insert_axis(Axis(0)).insert_axis(Axis(1)); // (1, 1, D, H, W)

    PreprocessData {
        input_tensor,
        original_shape: input_original_shape,
        ras_original_shape,
        resampled_shape: resampled_shape_ras,
        pad_size: [pad_d, pad_h, pad_w],
    }
}

pub fn get_orientation_info(header: &NiftiHeader) -> ([usize; 3], [i8; 3]) {
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

pub fn reorient_to_ras_with_info(mut data: Array3<f32>, codes: [usize; 3], signs: [i8; 3]) -> Array3<f32> {
    for j in 0..3 {
        if signs[j] == -1 {
            data.invert_axis(Axis(j));
        }
    }
    let mut perm = [0, 1, 2];
    for j in 0..3 {
        perm[codes[j]] = j;
    }
    data.permuted_axes(perm).to_owned().into_dimensionality::<Ix3>().unwrap()
}
