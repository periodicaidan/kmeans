kmeans
===

An implementation of k-means clustering algorithm

## How To Use It

1. Import `prelude`.
1. Implement the `DataPoint` trait for your data type.
1. Pass your data to the `kmeans` function.

```rust
use kmeans::prelude::*;

// `DataPoint` can only be implemented on types that are
// both `Clone` and `PartialEq`.
#[derive(Clone, Debug, Default, PartialEq)]
struct Color {
    r: f32,
    g: f32,
    b: f32
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }
}

impl DataPoint for Color {
    fn dist(&self, other: &Self) -> f64 {
        f64::sqrt(
            (&self.r - &other.r).powi(2) +
            (&self.g - &other.g).powi(2) +
            (&self.b - &other.b).powi(2)
        )   
    }
    
    fn mean(colors: &[Self]) -> Self {
        let mut sum = Color::default();
        for color in colors.iter() {
            sum.r += color.r;
            sum.g += color.g;
            sum.b += color.b;
        }   
        let n = colors.len() as f32;
        
        Color::new(sum.r / n, sum.g / n, sum.b / n)
    }
}

fn main() {
    let colors = vec![
        Color::new(0.5, 0.5, 0.5),
        Color::new(0.69, 0.42, 0.0),
        //...
    ];

    // Produces 3 clusters using the colors above
    let clusters = kmeans(3, colors.clone());

    println!("{:?}", Cluster::centroids(&clusters));
}
```