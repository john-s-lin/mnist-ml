pub mod common_data {

    pub struct CommonData {
        training_data: Option<Vec<f64>>,
        test_data: Option<Vec<f64>>,
        validation_data: Option<Vec<f64>>,
    }

    impl CommonData {
        pub fn new() -> Self {
            CommonData {
                training_data: None,
                test_data: None,
                validation_data: None,
            }
        }

        pub fn set_training_data(&mut self, training_data: Option<Vec>) {
            self.training_data = training_data;
        }
    }
}
