#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
#[macro_use] extern crate alloc;

pub mod types;

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use rand::prelude::*;
use types::*;
use core::ops::Add;

pub mod prelude {
    pub use super::{
        kmeans,
        types::{Cluster, DataPoint}
    };
}

/// Clustering algorithm using k-means++
pub fn kmeans<P: DataPoint>(k: usize, datapoints: Vec<P>) -> Vec<Cluster<P>> {
    // Initialize cluster means
    let mut clusters = _initialize_clusters(k, datapoints.clone());

    // Make the initial clusterings
    for point in datapoints.iter() {
        _cluster(point, &mut clusters);
    }

    // Store the current clustering for comparison later
    let mut prev_clusters = clusters.clone();

    // Recalculate the means based on the points in the cluster
    for cluster in clusters.iter_mut() {
        cluster.recalculate_centroid();
        cluster.points.clear();
    }
    // Cluster again
    for point in datapoints.iter() {
        _cluster(point, &mut clusters);
    }

    // Rinse, repeat; until the clusters cease to change
    while prev_clusters != clusters {
        prev_clusters = clusters.clone();
        for cluster in clusters.iter_mut() {
            cluster.recalculate_centroid();
            cluster.points.clear();
        }
        for point in datapoints.iter() {
            _cluster(point, &mut clusters);
        }
    }

    // Return the final clustering
    clusters
}

/// Initializes the clusters using an initialization algorithm based on [k-means++].
///
/// [k-means++]: https://en.wikipedia.org/wiki/K-means%2B%2B#Improved_initialization_algorithm
fn _initialize_clusters<P: DataPoint>(k: usize, mut datapoints: Vec<P>) -> Vec<Cluster<P>> {
    /// Returns one of the `centers` is closest to `point`.
    fn shortest_center_distance<P: DataPoint>(centers: &[P], point: &P) -> f64 {
        centers.iter()
            // calculate the distances between each center and `point`
            .map(|c| c.dist(point))
            // take the minimum of those distances
            .fold(f64::INFINITY, f64::min)
    }

    /// Selects a point using a weighted distribution based on `shortest_center_distance` squared.
    fn select_point(distribution: &[f64]) -> usize {
        // Generate the selection criterion for each point.
        // We'll generate a random number and select the point whose selection criterion is less
        // than that number, but whose following point's is greater than that number.
        // Like throwing a dart at a number line and seeing what range of values it falls in.
        let distr_sum = distribution.iter().fold(0.0, f64::add);
        let mut selection_criteria = Vec::with_capacity(distribution.len());
        for i in 0..distribution.len() {
            let sum = distribution[0..i].iter().fold(0.0, f64::add);
            selection_criteria.push(distribution[i] + sum);
        }
        let rn: f64 = thread_rng().gen_range(0.0, distr_sum);

        let mut selection = 0;
        for i in 1..(selection_criteria.len() - 1) {
            if rn > selection_criteria[i] {
                selection = i;
            }

            if rn <= selection_criteria[i] {
                break;
            }
        }

        selection
    }

    let mut clusters = Vec::with_capacity(k);
    // First centroid is selected with a uniform distribution
    let first_point = datapoints.remove(thread_rng().gen_range(0, datapoints.len()));
    let mut distribution: Vec<f64> = datapoints.iter()
        .map(|p| first_point.dist(p).powi(2))
        .collect();

    // Keep selecting unique points until we have `k` centroids
    while clusters.len() < k {
        let point = datapoints.remove(select_point(&distribution));
        clusters.push(Cluster::new(&point));
        let centroids = Cluster::centroids(&clusters);
        distribution = datapoints.iter()
            .map(|p| shortest_center_distance(&centroids, p).powi(2))
            .collect()
    }

    clusters
}

/// Assigns a point to the cluster whose centroid is closest
fn _cluster<P: DataPoint>(p: &P, clusters: &mut [Cluster<P>]) {
    let mut closest_cluster = 0;
    for c in 1..clusters.len() {
        if P::dist(p, &clusters[c].centroid) < P::dist(p, &clusters[closest_cluster].centroid) {
            closest_cluster = c;
        }
    }
    clusters[closest_cluster].points.push(p.clone());
}

#[cfg(test)]
mod test {
    #[cfg(feature = "alloc")]
    use alloc::{
        string::String,
        vec::Vec
    };

    use crate::prelude::*;

    #[test]
    fn float_clustering() {
        let points = vec![
            (1f64, 2f64), (1.0, 3.0), (2.0, 2.0), (2.0, 3.0), (2.0, 4.0),
            (3.0, 1.0), (3.0, 2.0), (3.0, 3.0), (3.0, 4.0), (4.0, 1.0), (4.0, 2.0), (4.0, 3.0),
            (4.0, 4.0), (5.0, 2.0), (5.0, 3.0),
            (6.0, 5.0), (6.0, 6.0), (6.0, 7.0), (7.0, 5.0), (7.0, 6.0), (7.0, 7.0), (7.0, 8.0),
            (8.0, 4.0), (8.0, 5.0), (8.0, 6.0), (8.0, 7.0), (8.0, 8.0), (9.0, 5.0), (9.0, 6.0),
            (9.0, 7.0), (9.0, 8.0), (10.0, 6.0), (10.0, 7.0)
        ];

        let clusters = kmeans(2, points);
        let centroids = Cluster::centroids(&clusters);

        assert!(centroids.contains(&(71.0/9.0, 113.0/18.0)));
        assert!(centroids.contains(&(46.0/15.0, 13.0/5.0)));
    }

    #[test]
    fn int_clustering() {
        let points = vec![
            (1u8, 2u8), (1, 3), (2, 2), (2, 3), (2, 4),
            (3, 1), (3, 2), (3, 3), (3, 4), (4, 1), (4, 2), (4, 3),
            (4, 4), (5, 2), (5, 3),
            (6, 5), (6, 6), (6, 7), (7, 5), (7, 6), (7, 7), (7, 8),
            (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (9, 5), (9, 6),
            (9, 7), (9, 8), (10, 6), (10, 7)
        ];

        let clusters = kmeans(2, points);
        let centroids = Cluster::centroids(&clusters);

        assert!(centroids.contains(&(7, 7)) || centroids.contains(&(8, 8)));
        assert!(centroids.contains(&(3, 3)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn file_test() {
        let file = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/test_data.txt")).unwrap();
        let data: Vec<(f64, f64, f64)> = file.trim().split("\n")
            .map(|line| {
                let strings: Vec<&str> = line.trim().split(" ").collect();
                (
                    strings[0].parse::<f64>().unwrap(),
                    strings[1].parse::<f64>().unwrap(),
                    strings[2].parse::<f64>().unwrap(),
                )
            })
            .collect();

        let clusters = kmeans(2, data.clone());
        let centroids = Cluster::centroids(&clusters);

        assert!(centroids.contains(&(9.98514851485149, 9.76534653465346, 10.132673267326735)));
        assert!(centroids.contains(&(2.5257425742574253, 2.5920792079207926, 2.7188118811881172)));
    }
}