pub mod data {
    pub struct Data {
        feature_vector: Option<Vec<u8>>,
        normalized_feature_vector: Option<Vec<f64>>,
        class_vector: Option<Vec<i32>>,
        label: u8,
        enum_label: i32,
        distance: f64,
    }

    impl Data {
        // NOTE: constructors and destructors are not implemented
        // since they are initialized immediately and dropped when out of scope       
    }
}
