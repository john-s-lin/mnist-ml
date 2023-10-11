pub mod data_handler {
    use std::collections::HashMap;

    const TRAIN_SET_PERCENT: f64 = 0.75;
    const TEST_SET_PERCENT: f64 = 0.20;
    const VALIDATION_SET_PERCENT: f64 = 0.05;

    pub struct DataHandler {
        data_array: Option<Vec<f64>>,
        training_data: Option<Vec<f64>>,
        test_data: Option<Vec<f64>>,
        validation_data: Option<Vec<f64>>,

        num_classes: i32,
        feature_vector_size: i64,

        class_labels: HashMap<u8, i32>,
        class_from_string: HashMap<String, i32>,
    }
}
