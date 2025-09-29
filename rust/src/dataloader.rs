use std::fs::File;
use std::path::Path;

use gpu_host::CudaMemSlice;

pub struct DataLoader {
    // ----------------------------------------------------------------------------
    // Hyperparameters
    // ----------------------------------------------------------------------------
    /// Batch size
    pub batch_size: usize,

    /// Sequence length
    pub seq_len: usize,

    // ----------------------------------------------------------------------------
    // Input handling and its state
    // ----------------------------------------------------------------------------
    /// File for tokens
    pub tokens_file: Option<File>,

    /// File size
    pub file_size: u64,

    /// Current position in the file
    pub current_position: u64,

    // ----------------------------------------------------------------------------
    // Output memory
    // ----------------------------------------------------------------------------
    /// Pointer to batch memory
    pub batch: Option<CudaMemSlice<i32>>,

    /// Pointer to input tokens
    pub inputs: Option<CudaMemSlice<i32>>,

    /// Pointer to target tokens
    pub targets: Option<CudaMemSlice<i32>>,

    // ----------------------------------------------------------------------------
    // Convenience variables
    // ----------------------------------------------------------------------------
    /// Number of batches
    pub num_batches: usize,
}

impl DataLoader {
    /// Creates a new DataLoader instance.
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the tokens file.
    /// * `batch` - Batch size.
    /// * `T` - Sequence length.
    ///
    /// # Returns
    ///
    /// A new `DataLoader` instance.
    pub fn new(filename: &Path, batch_size: usize, seq_len: usize) -> Self {
        let mut loader = DataLoader {
            batch_size,
            seq_len,
            tokens_file: None,
            file_size: 0,
            current_position: 0,
            batch: None,
            inputs: None,
            targets: None,
            num_batches: 0,
        };

        loader.tokens_file = match File::open(filename) {
            Ok(file) => Some(file),
            Err(_) => {
                panic!("Error opening tokens file");
            }
        };
        unimplemented!()
    }

    /// Resets the DataLoader to start from the beginning of the file.
    pub fn reset(&mut self) {
        self.current_position = 0;
    }

    /// Loads the next batch of data into the DataLoader's memory.
    pub fn next_batch(&mut self) {
        unimplemented!()
    }
}
