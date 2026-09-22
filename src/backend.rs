pub trait TensorBackend: Send + Sync {
    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32>;
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;
    fn rope(&self, x: &mut [f32], pos: usize, head_dim: usize, theta: f32);
}

pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    fn rms_norm(&self, x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(w.iter()).map(|(a, b)| a * inv * b).collect()
    }

    fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        // a: [m, k], b: [k, n] row-major -> [m, n]
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    out[i * n + j] += av * b[p * n + j];
                }
            }
        }
        out
    }

    fn rope(&self, x: &mut [f32], pos: usize, head_dim: usize, theta: f32) {
        for i in (0..x.len()).step_by(head_dim) {
            for j in (0..head_dim).step_by(2) {
                let freq = 1.0 / theta.powf(j as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                let x0 = x[i + j];
                let x1 = x[i + j + 1];
                x[i + j] = x0 * c - x1 * s;
                x[i + j + 1] = x0 * s + x1 * c;
            }
        }
    }
}
