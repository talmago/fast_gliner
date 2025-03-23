//! First step of span decoding (in token mode)

use composable::Composable;
use crate::util::result::Result;
use crate::util::math::sigmoid;
use crate::text::span::Span;
use crate::model::pipeline::context::EntityContext;
use crate::model::output::tensors::TensorOutput;
use super::SpanOutput;


const TENSOR_LOGITS: &str = "logits";


/// Decoding method for token mode.
/// 
/// From the related (GLiNER multi-task) [paper](https://arxiv.org/pdf/2406.12925v1): 
/// 
/// > The function `ϕi(k,c)` is the inside score for the token at position `k` within the span, 
/// > indicating the likelihood of the token being part of the entity class `c`. By averaging
/// > these scores over the span, we obtain a measure of how well the span fits the token class `c`.
/// 
/// Rq: greedy search must be applied the same way as in span mode (shall be called as a subsequent 
/// step in the pipeline).
pub struct TensorsToDecoded {
    threshold: f32,
}

impl TensorsToDecoded {
    pub fn new(threshold: f32) -> Self {
        Self { 
            threshold,
        }
    }

    pub fn outputs() -> [&'static str; 1] {
        [TENSOR_LOGITS]
    }

    fn decode(&self, input: &TensorOutput) -> Result<Vec<Vec<Span>>> {
        // prepare output vector
        let batch_size = input.context.texts.len();
        let mut result: Vec<Vec<Span>> = std::iter::repeat_with(Vec::new).take(batch_size).collect();

        // look for logits and check its shape
        let logits = input.tensors.get(TENSOR_LOGITS).ok_or("logits not found in model output")?;
        self.check_shape(logits.shape()?, &input.context)?;
    
        // extract the actual array
        let array  = logits.try_extract_tensor::<f32>()?;
        //println!("{:?}", array.map(|x| crate::util::math::sigmoid(*x)));

        // iterate over sequences
        for sequence_id in 0..batch_size {
            // get a slice for the current sequence (2nd dimension)
            let scores = array.slice(ndarray::s![.., sequence_id, .., ..]);

            // get slices for start, end, and inside scores (1st dimension)
            let scores_start = scores.slice(ndarray::s![0, .., ..]);
            let scores_end = scores.slice(ndarray::s![1, .., ..]);
            let scores_inside = scores.slice(ndarray::s![2, .., ..]);            

            // generate all possible spans and iterate over them
            for span in self.generate_spans(&scores_start, &scores_end) {
                // compute score
                let score = self.compute_span_score(span, &scores_inside);
                // reject span if score is below threshold
                if score < self.threshold {
                    continue
                }                
                // create actual span
                let (start_token, end_token, class) = span;
                let span = input.context.create_span(sequence_id, start_token, end_token, class, score)?;
                result.get_mut(sequence_id).unwrap().push(span);
            }
        }
        Ok(result)
    }


    /// Generates all possible `(i,j,c)` spans where:
    /// * `i <= j`
    /// * `score(i) >= threshold`
    /// * `score(j) >= threshold`.
    /// * `c` == `class(i) == class(j)`
    fn generate_spans(&self, scores_start: &ndarray::ArrayView2::<f32>, scores_end: &ndarray::ArrayView2::<f32>) -> Vec<(usize, usize, usize)> {
        assert!(scores_start.dim() == scores_end.dim());
        let (num_tokens, num_classes) = scores_start.dim();
        let mut result = Vec::new();
        for class in 0..num_classes {
            for start in 0..num_tokens {            
                let score_start = sigmoid(*scores_start.get((start, class)).unwrap());
                if score_start < self.threshold {
                    continue
                }
                for end in start..num_tokens {
                    let score_end = sigmoid(*scores_end.get((end, class)).unwrap());
                    if score_end < self.threshold {
                        continue
                    }
                    result.push((start, end, class));
                }
            }
        }
        result 
    }


    /// Computes the score of a span, defined as the mean of the inside scores (see above).
    /// Spans with one or more inside scores below the threshold will return a zero score, 
    /// since they should be discarded.
    fn compute_span_score(&self, span: (usize, usize, usize), scores_inside: &ndarray::ArrayView2<f32>) -> f32 {
        let (start, end, class) = span;
        assert!(end >= start);
        let mut sum = 0f32;
        for i in start..end+1 {
            let score_inside = sigmoid(*scores_inside.get((i, class)).unwrap());
            if score_inside < self.threshold {
                return 0.;
            }
            sum += score_inside;
        }
        sum / ((end - start + 1) as f32)
    }
    

    /// Checks coherence of the output shape.
    /// Expected shape is (3, batch_size, num_words, num_classes).
    /// The first dimension is related to `start`, `end` and `inside` positions in that order.
    fn check_shape(&self, actual_shape: Vec<i64>, context: &EntityContext) -> Result<()> {
        let expected_shape = vec![3, context.texts.len() as i64, context.num_words as i64, context.entities.len() as i64];
        if actual_shape != expected_shape {
            Err("unexpected logits shape".into())
        }
        else {
            Ok(())
        }
    }

}

impl Composable<TensorOutput<'_>, SpanOutput> for TensorsToDecoded {
    fn apply(&self, input: TensorOutput) -> Result<SpanOutput> {        
        let decoded = self.decode(&input)?;
        Ok(SpanOutput::new(input.context.texts, input.context.entities, decoded))
    }
}