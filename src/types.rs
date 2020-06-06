#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::iter::Sum;
use core::convert::TryFrom;
use core::fmt::Debug;

/// A representation of a point in a data space.
pub trait DataPoint: Clone + PartialEq {
    /// Calculates the distance between two points
    fn dist(&self, other: &Self) -> f64;

    /// Calculates the mean of a slice of points
    fn mean(ps: &[Self]) -> Self;
}

/// A clustering of `points` around a `centroid`
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Cluster<P: DataPoint> {
    pub centroid: P,
    pub points: Vec<P>
}

impl<P: DataPoint> Cluster<P> {
    pub fn new(centroid: &P) -> Self {
        Self { centroid: centroid.clone(), points: vec![] }
    }

    pub fn from_intermediate(intermediate: &IntermediateCluster<P>, points: &[P]) -> Self {
        Self {
            centroid: intermediate.centroid.clone(),
            points: intermediate.point_indices.iter().map(|i| points[*i].clone()).collect()
        }
    }

    pub fn recalculate_centroid(&mut self) {
        self.centroid = P::mean(&self.points);
    }

    pub fn centroids(cs: &[Self]) -> Vec<P> {
        cs.iter().map(|c| c.centroid.clone()).collect()
    }
}

pub struct IntermediateCluster<P: DataPoint> {
    pub centroid: P,
    pub point_indices: Vec<usize>
}

impl<P: DataPoint> IntermediateCluster<P> {
    pub fn new(centroid: &P) -> Self {
        Self { centroid: centroid.clone(), point_indices: vec![] }
    }
}

/*** IMPLS FOR COMMON POINT REPRESENTATIONS ***/

macro_rules! impl_float_2d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 - other.0).powi(2) as f64 +
                    (self.1 - other.1).powi(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0.0, 0.0), |acc, next| (acc.0 + next.0, acc.1 + next.1));

                (sum.0 / ps.len() as $T, sum.1 / ps.len() as $T)
            }
        }
    };
}

macro_rules! impl_unsigned_2d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                fn abs_sub(a: $T, b: $T) -> $T {
                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                }

                f64::sqrt(
                    abs_sub(self.0, other.0).pow(2) as f64 +
                    abs_sub(self.1, other.1).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0), |acc, next| (acc.0 + next.0 as usize, acc.1 + next.0 as usize));

                (
                    <$T>::try_from(sum.0 / ps.len()).unwrap(),
                    <$T>::try_from(sum.1 / ps.len()).unwrap()
                )
            }
        }
    };
}

macro_rules! impl_signed_2d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 as isize - other.0 as isize).pow(2) as f64 +
                    (self.1 as isize - other.1 as isize).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0), |acc, next| (acc.0 + next.0 as isize, acc.1 + next.0 as isize));

                (
                    <$T>::try_from(sum.0 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.1 / ps.len() as isize).unwrap()
                )
            }
        }
    };
}

macro_rules! impl_float_3d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 - other.0).powi(2) as f64 +
                    (self.1 - other.1).powi(2) as f64 +
                    (self.2 - other.2).powi(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0.0, 0.0, 0.0), |acc, next| (acc.0 + next.0, acc.1 + next.1, acc.2 + next.2));

                (sum.0 / ps.len() as $T, sum.1 / ps.len() as $T, sum.2 / ps.len() as $T)
            }
        }
    };
}

macro_rules! impl_unsigned_3d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                fn abs_sub(a: $T, b: $T) -> $T {
                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                }

                f64::sqrt(
                    abs_sub(self.0, other.0).pow(2) as f64 +
                    abs_sub(self.1, other.1).pow(2) as f64 +
                    abs_sub(self.2, other.2).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0, 0), |acc, next| (acc.0 + next.0 as usize, acc.1 + next.0 as usize, acc.2 + next.2 as usize));

                (
                    <$T>::try_from(sum.0 / ps.len()).unwrap(),
                    <$T>::try_from(sum.1 / ps.len()).unwrap(),
                    <$T>::try_from(sum.2 / ps.len()).unwrap()
                )
            }
        }
    };
}

macro_rules! impl_signed_3d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 as isize - other.0 as isize).pow(2) as f64 +
                    (self.1 as isize - other.1 as isize).pow(2) as f64 +
                    (self.2 as isize - other.2 as isize).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0, 0), |acc, next| (acc.0 + next.0 as isize, acc.1 + next.0 as isize, acc.2 + next.2 as isize));

                (
                    <$T>::try_from(sum.0 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.1 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.2 / ps.len() as isize).unwrap()
                )
            }
        }
    };
}

macro_rules! impl_float_4d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 - other.0).powi(2) as f64 +
                    (self.1 - other.1).powi(2) as f64 +
                    (self.2 - other.2).powi(2) as f64 +
                    (self.3 - other.3).powi(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0.0, 0.0, 0.0, 0.0), |acc, next| (acc.0 + next.0, acc.1 + next.1, acc.2 + next.2, acc.3 + next.3));

                (sum.0 / ps.len() as $T, sum.1 / ps.len() as $T, sum.2 / ps.len() as $T, sum.3 / ps.len() as $T)
            }
        }
    };
}

macro_rules! impl_unsigned_4d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                fn abs_sub(a: $T, b: $T) -> $T {
                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                }

                f64::sqrt(
                    abs_sub(self.0, other.0).pow(2) as f64 +
                    abs_sub(self.1, other.1).pow(2) as f64 +
                    abs_sub(self.2, other.2).pow(2) as f64 +
                    abs_sub(self.3, other.3).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0, 0, 0), |acc, next| (acc.0 + next.0 as usize, acc.1 + next.0 as usize, acc.2 + next.2 as usize, acc.3 + next.3 as usize));

                (
                    <$T>::try_from(sum.0 / ps.len()).unwrap(),
                    <$T>::try_from(sum.1 / ps.len()).unwrap(),
                    <$T>::try_from(sum.2 / ps.len()).unwrap(),
                    <$T>::try_from(sum.3 / ps.len()).unwrap()
                )
            }
        }
    };
}

macro_rules! impl_signed_4d_data_point {
    ($T:ty) => {
        impl DataPoint for ($T, $T, $T, $T) {
            fn dist(&self, other: &Self) -> f64 {
                f64::sqrt(
                    (self.0 as isize - other.0 as isize).pow(2) as f64 +
                    (self.1 as isize - other.1 as isize).pow(2) as f64 +
                    (self.2 as isize - other.2 as isize).pow(2) as f64 +
                    (self.3 as isize - other.3 as isize).pow(2) as f64
                )
            }

            fn mean(ps: &[Self]) -> Self {
                let sum = ps.iter()
                    .fold((0, 0, 0, 0), |acc, next| (acc.0 + next.0 as isize, acc.1 + next.0 as isize, acc.2 + next.2 as isize, acc.3 + next.3 as isize));

                (
                    <$T>::try_from(sum.0 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.1 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.2 / ps.len() as isize).unwrap(),
                    <$T>::try_from(sum.3 / ps.len() as isize).unwrap()
                )
            }
        }
    };
}

impl_float_2d_data_point!(f32);
impl_float_2d_data_point!(f64);
impl_unsigned_2d_data_point!(u8);
impl_unsigned_2d_data_point!(u16);
impl_unsigned_2d_data_point!(u32);
impl_unsigned_2d_data_point!(u64);
impl_unsigned_2d_data_point!(usize);
impl_signed_2d_data_point!(i8);
impl_signed_2d_data_point!(i16);
impl_signed_2d_data_point!(i32);
impl_signed_2d_data_point!(i64);
impl_signed_2d_data_point!(isize);

impl_float_3d_data_point!(f32);
impl_float_3d_data_point!(f64);
impl_unsigned_3d_data_point!(u8);
impl_unsigned_3d_data_point!(u16);
impl_unsigned_3d_data_point!(u32);
impl_unsigned_3d_data_point!(u64);
impl_unsigned_3d_data_point!(usize);
impl_signed_3d_data_point!(i8);
impl_signed_3d_data_point!(i16);
impl_signed_3d_data_point!(i32);
impl_signed_3d_data_point!(i64);
impl_signed_3d_data_point!(isize);

impl_float_4d_data_point!(f32);
impl_float_4d_data_point!(f64);
impl_unsigned_4d_data_point!(u8);
impl_unsigned_4d_data_point!(u16);
impl_unsigned_4d_data_point!(u32);
impl_unsigned_4d_data_point!(u64);
impl_unsigned_4d_data_point!(usize);
impl_signed_4d_data_point!(i8);
impl_signed_4d_data_point!(i16);
impl_signed_4d_data_point!(i32);
impl_signed_4d_data_point!(i64);
impl_signed_4d_data_point!(isize);