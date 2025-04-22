# GLiNER ONNX Models Properties

## Model Properties in Span Mode

### Input

* `input_ids` : `int64[batch_size,sequence_length]`
* `attention_mask` : `int64[batch_size,sequence_length]`
* `words_mask` : `int64[batch_size,sequence_length]`
* `text_lengths` : `int64[batch_size,value]`
* `span_idx` : `int64[batch_size,num_spans,idx]`
* `span_mask` : `boolean[batch_size,num_spans]`

### Output

* `logits` : `float32[batch_size,sequence_length,num_spans,num_classes]`


## Model Properties in Token Mode

### Input

* `input_ids`: `int64[batch_size,sequence_length]`
* `attention_mask`: `int64[batch_size,sequence_length]`
* `words_mask`: `int64[batch_size,sequence_length]`
* `text_lengths: int64[batch_size,value]`

### Output

* `logits`: `float32[position,batch_size,sequence_length,num_classes]`


