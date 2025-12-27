mod preprocess;
mod postprocess;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::prelude::*;
use nifti::{IntoNdArray, NiftiObject, ReaderOptions};
use nifti::writer::WriterOptions;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: Option<PathBuf>,

    #[arg(short, long)]
    model: PathBuf,

    #[arg(long, default_value_t = true)]
    fast: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // 1. Load Model
    println!("Loading model from: {:?}", args.model);
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(8)?
        .commit_from_file(&args.model)?;


    // 2. Load Input NIfTI
    println!("Processing: {:?}", args.input);
    let start_total = Instant::now();
    
    let obj = ReaderOptions::new().read_file(&args.input)?;
    let header = obj.header().clone();
    let data: Array3<f32> = obj.into_volume().into_ndarray::<f32>()?.into_dimensionality::<Ix3>()?;
    let spacing = [header.pixdim[1], header.pixdim[2], header.pixdim[3]];
    
    println!("Input statistics:");
    println!("  Shape: {:?}", data.shape());
    println!("  Mean:  {:.4}", data.mean().unwrap_or(0.0));
    println!("  Range: [{:.2}, {:.2}]", 
             data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

    if !header.scl_slope.is_nan() && header.scl_slope != 0.0 && header.scl_slope != 1.0 || 
       !header.scl_inter.is_nan() && header.scl_inter != 0.0 {
        println!("  NIfTI header scaling present: slope={:.4}, inter={:.4} (already applied by reader)", 
                 header.scl_slope, header.scl_inter);
    }

    // 3. Preprocessing
    let start_pre = Instant::now();
    let pre_data = preprocess::preprocess(data, spacing, &header);
    let pre_time = start_pre.elapsed();
    println!("  Preprocessing: {:?}", pre_time);

    // Save a slice of the input tensor for comparison
    {
        use std::fs::File;
        use std::io::Write;
        let tensor = &pre_data.input_tensor;
        let d = tensor.shape()[2];
        let h = tensor.shape()[3];
        let w = tensor.shape()[4];
        let mid_z = 60; // Use fixed slice for comparison
        let mut f = File::create("rust_slice.txt")?;
        for y in 0..h {
            for x in 0..w {
                write!(f, "{:.4} ", tensor[[0, 0, mid_z, y, x]])?;
            }
            writeln!(f)?;
        }
        println!("  Saved preprocessed slice to rust_slice.txt (shape: [{}, {}, {}], mid_z: {})", d, h, w, mid_z);
    }

    // 4. Inference
    let start_inf = Instant::now();

    // Bridge: Array5 (0.16) to ort-compatible Tensor (creates a copy due to version mismatch)
    let input_slice = pre_data.input_tensor.as_slice().context("Failed to get input slice")?;
    let input_shape: Vec<i64> = pre_data.input_tensor.shape().iter().map(|&d| d as i64).collect();
    // Using (Vec<i64>, Vec<f32>) to satisfy OwnedTensorArrayData
    let input_tensor = ort::value::Tensor::from_array((input_shape, input_slice.to_vec()))?;
    
    let outputs = session.run(ort::inputs![input_tensor])?;
    let inf_time = start_inf.elapsed();
    println!("  Inference:     {:?}", inf_time);

    // 5. Postprocessing
    let start_post = Instant::now();
    let prediction_val = &outputs[0];
    let (shape, data) = prediction_val.try_extract_tensor::<f32>()?;
    
    // Convert to our ndarray (0.16)
    let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let prediction_view = ArrayView::from_shape(shape_vec, data)?;
    let prediction_owned = prediction_view.to_owned().into_dimensionality::<Ix5>()?;
    
    let seg_final = postprocess::postprocess(
        prediction_owned, 
        pre_data.resampled_shape, 
        pre_data.ras_original_shape,
        pre_data.original_shape, 
        pre_data.pad_size,
        &header
    );
    let post_time = start_post.elapsed();
    println!("  Postprocessing: {:?}", post_time);

    // 6. Save Result
    let output_path = args.output.unwrap_or_else(|| {
        let mut p = args.input.clone();
        let stem = p.file_stem().unwrap().to_str().unwrap();
        p.set_file_name(format!("{}_seg_rust.nii.gz", stem));
        p
    });
    
    println!("  Saving to: {:?}", output_path);
    WriterOptions::new(&output_path)
        .reference_header(&header)
        .write_nifti(&seg_final)?;

    println!("Total time: {:?}", start_total.elapsed());



    Ok(())
}
