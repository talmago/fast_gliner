#set heading(numbering: "1.")
#show link: underline

#place(
  top + center, float: true, scope: "parent",
  text(1.8em, weight: "bold")[Matrix-Level Documentation of \ `gline-rs` Processing Steps]  
)

#place(
  top + center, float: true, scope: "parent",
  text[Frédérik Bilhaut]
)

#v(20pt)

This documents aims at providing a matrix-level description of the pipeline needed for GLiNER inferences, as implemented by #link("https://github.com/fbilhaut/gline-rs")[`gline-rs`].

Concrete examples are provided for each step, all of which build on the input given in the first one.

#v(20pt)
#outline(depth: 2)

#pagebreak()
= Pre-Processing (Common)

== Text Input

This is the "user" input for the whole processing.

=== Source Code

- Related struct: `gliner::model::input::text::TextInput`

=== Format

- $n$: number of input texts
- $k$: number of entity class labels
- $I$: sequence of input texts matrix of type `string` and size $n$
- $E$: entity class labels matrix, of type `string` and size $k$

#align(left, block($ 
I =
  mat(delim: "[",
  "text"_1;
  "text"_2;
  dots.v;
  "text"_n
)
$))

#align(left, block($
E = mat(delim: "[",
  "label"_1;
  "label"_2;
  dots.v;
  "label"_k
)
$))


=== Example

#align(left, block($
I =
  mat(delim: "[",
  &"\"My name is James Bond\"";
  &"\"I like to drive my Aston Martin\"";  
)
$))

#align(left, block($
E = mat(delim: "[",
  &"\"movie character\"";
  &"\"vehicle\"";
)
$))

#pagebreak()
== Word-Level Tokenization

=== Transformation

#align(left, block($(I, E) -> (T, E)$))

=== Source Code

- Struct: `gliner::model::input::tokenized::TokenizedInput`
- Transformation: `gliner::model::input::prompt::RawToTokenized`

=== Format

- $n, k$:  same as before
- $T$: sequence of sequence of tokenized input texts, of type `string` and size $n$
- $E$: same as before

#align(left, block($
T = mat(delim: "[",
  mat(delim: "[", "token"_"1,1", "token"_"1,2", dots);
  mat(delim: "[", "token"_"2,1", "token"_"2,2", dots);
  dots.v;
  mat(delim: "[", "token"_"n,1", "token"_"n,2", dots);
)
$))

=== Example

#align(left, block($
T = mat(delim: "[",
  &mat(delim: "[", "\"My\"" "\"name\"", "\"is\"", "\"James\"", "\"Bond\"");  
  &mat(delim: "[", "\"I\"" "\"like\"", "\"to\"", "\"drive\"", "\"my\"", "\"Aston\"", "\"Martin\"");  
)
$))

#pagebreak()
== Prompt Preparation

Prepared prompts, appending entity and text tokens.

=== Transformation

#align(left, block($(T, E) -> P$))

=== Source Code

- Struct: `gliner::model::input::prompt::PromptInput`
- Transformation from `TokenizedInput`: `gliner::model::input::prompt::TokenizedToPrompt`

=== Format

#align(left, block($
P = mat(delim: "[",
  mat(delim: "[", "<<ENT>>", "label"_"1,1", "<<ENT>>", "label"_"1,2", dots, "<<SEP>>", "token"_"1,1", , "token"_"1,2", dots);
  mat(delim: "[", "<<ENT>>", "label"_"2,1", "<<ENT>>", "label"_"2,2", dots, "<<SEP>>", "token"_"2,1", , "token"_"2,2", dots);
  dots.v ;
  mat(delim: "[", "<<ENT>>", "label"_"n,1", "<<ENT>>", "label"_"n,2", dots, "<<SEP>>", "token"_"n,1", , "token"_"n,2", dots)
)
$))

=== Example

#align(left, block($
P = mat(delim: "[",
  &mat(delim: "[", "<<ENT>>", "\"movie character\"", "<<ENT>>", "\"vehicle\"", dots, "<<SEP>>", "\"My\"", "\"name\"", "\"is\"", "\"James\"", "\"Bond\"");
  &mat(delim: "[", "<<ENT>>", "\"movie character\"", "<<ENT>>", "\"vehicle\"", dots, "<<SEP>>", "\"I\"", "\"like\"", "\"to\"", "\"drive\"", "\"my\"", "\"Austin\"", "\"Martin\"");
)
$))


#pagebreak()
== Prompt Encoding (Sub-Word Tokenization)

=== Transformation

#align(left, block($P -> (I, A, W, L)$))

=== Source Code

- Struct: `gliner::model::input::encoded::EncodedPrompt`
- Transformation: `gliner::model::input::encoded::PromptsToEncoded`

=== Format

#let ststart(x) = text(fill: green, $#x$)
#let stend(x) = text(fill: red, $#x$)
#let stent(x) = text(fill: orange, $#x$)

- k: maximum number of sub-word tokens within a sequence, adding start ($ststart(1)$) and end ($stend(2)$) tokens
- I: encoded prompts of type `i64` and shape $(n*k)$
- A: attention masks of type `i64` and shape $(n*k)$
- W: word masks of type `i64` and shape $(n*k)$
- L: text lengths of type `i64` and shape $(n*1)$

#align(left, block($
I = mat(
  "token_id"_"1,1", "token_id"_"1,2", dots, "token_id"_"1,k" ;
  "token_id"_"2,1", "token_id"_"2,2", dots, "token_id"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "token_id"_"n,1", "token_id"_"n,2", dots, "token_id"_"n,k" ;  
)
$))

#align(left, block(
$
A = mat(
  "attn_mask"_"1,1", "attn_mask"_"1,2", dots, "attn_mask"_"1,k" ;
  "attn_mask"_"2,1", "attn_mask"_"2,2", dots, "attn_mask"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "attn_mask"_"n,1", "attn_mask"_"n,2", dots, "attn_mask"_"n,k" ;  
)
$))

#align(left, block($
W = mat(
  "word_mask"_"1,1", "word_mask"_"1,2", dots, "word_mask"_"1,k" ;
  "word_mask"_"2,1", "word_mask"_"2,2", dots, "word_mask"_"2,k" ;
  dots.v, dots.v, dots.down, dots.v ;
  "word_mask"_"n,1", "word_mask"_"n,2", dots, "word_mask"_"n,k" ;  
)
$))

#align(left, block($
L = mat(
  "l"_"1"; 
  dots.v;
  "l"_"n";
)
$))

=== Example

#align(left, block($
I = mat(
  ststart(1), stent(128002), 1421, 1470, stent(128002), 1508, stent(128003), 573, 601, 269, 1749, 8728, stend(2), 0, 0;
  ststart(1), stent(128002), 1421, 1470, stent(128002), 1508, stent(128003), 273, 334, 264, 1168, 312, 20844, 2963, stend(2);
)
$))

#align(left, block($
A = mat(
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0;
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
)
$))

#align(left, block($
W = mat(
  0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0, 0;
  0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 0;
)
$))

#align(left, block($
L = mat(
  5;
  7
)
$))

#pagebreak()
= Pre-Processing (Span Mode)

Downstream of the aforementioned steps.

== Span Preparation

=== Transformation

#align(left, block($(I, A, W, L) -> (I, A, W, L, S_I, S_M)$))

=== Format

- $n, k, I, A, W, L$: same as before.
- $s$: maximum possible number of spans for one sequence
- $S_I$: span offsets, of type `i64` and shape $(n*s*2)$
- $S_M$: span masks, of type `bool` and shape $(n*s)$

#align(left, block($
S_I = mat(
  mat("start"_"1,1", "end"_"1,1"), mat("start"_"1,2", "end"_"1,2"), dots, mat("start"_"1,s", "end"_"1,s");
  mat("start"_"2,1", "end"_"2,1"), mat("start"_"2,2", "end"_"2,2"), dots, mat("start"_"2,s", "end"_"2,s");
  dots.v, dots.v, dots.down, dots.v;
  mat("start"_"n,1", "end"_"n,1"), mat("start"_"n,2", "end"_"n,2"), dots, mat("start"_"n,s", "end"_"n,s");
)
$))

#align(left, block($
S_M = mat(
  "span_mask"_"1,1", "span_mask"_"1,2", dots, "span_mask"_"1,s";
  "span_mask"_"2,1", "span_mask"_"2,2", dots, "span_mask"_"2,s";
  dots.v, dots.v, dots.down, dots.v;
  "span_mask"_"n,1", "span_mask"_"n,2", dots, "span_mask"_"n,s";
)
$))

=== Example

Note: for readability purposes, inside matrices are split into rows (one per token) but they are actually in one dimension $s$ (see format above).

#align(left, block($
S_I = mat(
  mat(
    mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(0, 4), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), ⏎; 
    mat(1, 1), mat(1, 2), mat(1, 3), mat(1, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(2, 2), mat(2, 3), mat(2, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(3, 3), mat(3, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(4, 4), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots, dots
  )
  ;
  mat(
    mat(0, 0), mat(0, 1), mat(0, 2), mat(0, 3), mat(0, 4), mat(0, 5), mat(0, 6), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), ⏎;
    mat(1, 1), mat(1, 2), mat(1, 3), mat(1, 4), mat(1, 5), mat(1, 6), mat(0, 0), dots, dots, dots, dots, dots, ⏎;
    mat(2, 2), mat(2, 3), mat(2, 4), mat(2, 5), mat(2, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, ⏎;
    mat(3, 3), mat(3, 4), mat(3, 5), mat(3, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(4, 4), mat(4, 5), mat(4, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(5, 5), mat(5, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, ⏎;
    mat(6, 6), mat(0, 0), dots, dots, dots, dots, dots, dots, dots, dots, dots, dots
  )
)
$))

#align(left, block($
S_M = mat(
  mat(
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ⏎; 
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  )
  ;
  mat(
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ⏎;
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0; 
  )
)
$))

#pagebreak()
= Pre-Processing (Token Mode)

Nothing to be done beside the common steps.

#pagebreak()
= Post-Processing (Span Mode)

== Logits Output

=== Source Code

- Struct: `gliner::model::output::TensorOutput`

=== Format

- $n$: number of text sequences
- $w$: maximum number of tokens in one sequence
- $s$: maximum number of possible spans for one token (seee above)
- $k$: number of entity labels
- $O$: logits output, of type `f32` and shape $(n*w*s*k)$
- $v_"n,w,s,k"$: raw model output for sequence $n$, token $w$, span $s$ and label $k$.

#align(left, block($
O = mat(
  mat(
    mat(
      mat("v"_"1,1,1,1", dots, "v"_"1,1,1,k");
      dots.v;
      mat("v"_"1,1,s,1", dots, "v"_"1,1,s,k");
    ), 
    dots,
    mat(
      mat("v"_"1,w,1,1", dots, "v"_"1,w,1,k");
      dots.v;
      mat("v"_"1,w,s,1", dots, "v"_"1,w,s,k");
    )
  );
  dots.v;
  mat(
    mat(
      mat("v"_"n,1,1,1", dots, "v"_"n,1,1,k");
      dots.v;
      mat("v"_"n,1,s,1", dots, "v"_"n,1,s,k");
    ), 
    dots,
    mat(
      mat("v"_"n,w,1,1", dots, "v"_"n,w,1,k");
      dots.v;
      mat("v"_"n,w,s,1", dots, "v"_"n,w,s,k");
    )
  )
)
$))


=== Example

In this case $s=12$. For readability purposes, the raw values are "sigmoided" ($S(x)= 1/(1+e^(-x))$) and then "ReLUed" with a threshold $t=0.5$.


#align(left, block($
O_"S,t" = mat(
  mat(
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(bold(0.89), 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);      
  );
  mat(
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);      
      mat(0, 0), mat(0, bold(0.96)), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);
      mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0), mat(0, 0);      
  );
)
$))

Which means:
- In the 1st sequence, the span starting with the 4th token and ending with the 5th one has a probability of 0.89 to match the 1st entity class.
- In the 2nd sequence, the span starting with the 6th token and ending with the 7th one has a probability of 0.96 to match the second 2nd class.

#pagebreak()
== Span Decoding

=== Transformation

$(O, L) -> S$

=== Source Code

- Struct: `gliner::model::output::decoded::SpanOutput`
- Transformation: `gliner::model::output::decoded::span::TensorsToDecoded`

=== Format

- $t$: threshold
- $n$: number of input sequences
- $L$: text lengths as defined before
- $S$: sequence of spans $(i,j,k,p)$ where: 
  - $i$ is the index of the first token of sequence $m$ with $i<j$ and $i<L(m)$ 
  - $j$ is the index of the last token with the same constraints as $i$
  - $k$ is the entity class, 
  - $p$ is the probability for class $k$ with $p>=t$

#align(left, block($ 
S =
  mat(delim: "[",
    mat(delim: "[", (i_"1,1", j_"1,1", k_"1,1", p_"1,1"), (i_"1,2", j_"1,2", k_"1,2", p_"1,2"), dots);
    dots.v;
    mat(delim: "[", (i_"n,1", j_"n,1", k_"n,1", p_"n,1"), (i_"n,2", j_"n,2", k_"n,2", p_"n,2"), dots);
)
$))

=== Example

#align(left, block($ 
S =
  mat(delim: "[",
    mat(delim: "[", (4, 5, 1, 0.89));
    mat(delim: "[", (6, 7, 2, 0.96));
)
$))

#pagebreak()
= Post-Processing (Token Mode)

== Logits Output

=== Source Code

- Struct: `gliner::model::output::TensorOutput`

=== Format


- $n$: number of text sequences
- $w$: maximum number of tokens in one sequence
- $k$: number of entity labels
- $O$: logits output, of type `f32` and shape $(3*n*w*k)$ with:
  - $s_"n,w,k"$: raw model output for a start token $w$ in sequence $n$ and label $k$.
  - $e_"n,w,k"$: raw model output for an end token $w$ in sequence $n$ and label $k$.
  - $i_"n,w,k"$: raw model output for an inside token $w$ in sequence $n$ and label $k$.

#align(left, block($
O = mat(
  mat(
    mat(
      "s"_"1,1,1", dots, "s"_"1,1,k";
      dots.v, dots.down, dots.v;
      "s"_"1,w,1", dots, "s"_"1,w,k";
    ), 
    dots,
    mat(
      "s"_"n,1,1", dots, "s"_"n,1,k";
      dots.v, dots.down, dots.v;
      "s"_"n,w,1", dots, "s"_"n,w,k";
    ), 
  );
  mat(
    mat(
      "e"_"1,1,1", dots, "e"_"1,1,k";
      dots.v, dots.down, dots.v;
      "e"_"1,w,1", dots, "e"_"1,w,k";
    ), 
    dots,
    mat(
      "e"_"n,1,1", dots, "e"_"n,1,k";
      dots.v, dots.down, dots.v;
      "e"_"n,w,1", dots, "e"_"n,w,k";
    ), 
  );
  mat(
    mat(
      "i"_"1,1,1", dots, "i"_"1,1,k";
      dots.v, dots.down, dots.v;
      "i"_"1,w,1", dots, "i"_"1,w,k";
    ), 
    dots,
    mat(
      "i"_"n,1,1", dots, "i"_"n,1,k";
      dots.v, dots.down, dots.v;
      "i"_"n,w,1", dots, "i"_"n,w,k";
    ), 
  );
)
$))

=== Example

For readability purposes, the raw values are "sigmoided" ($S(x)= 1/(1+e^(-x))$) and then "ReLUed" with a threshold $t=0.5$.

#align(left, block($
O_"S,t" = mat(
  mat(
    mat(0, 0; 0, 0; 0, 0; bold(0.97), 0; 0, 0; 0, 0; 0, 0),
    mat(0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, bold(0.99); 0, 0), 
  );
  mat(
    mat(0, 0; 0, 0; 0, 0; 0, 0; bold(0.96), 0; 0, 0; 0, 0),
    mat(0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, bold(0.97)),
  );
  mat(
    mat(0, 0; 0, 0; 0, 0; bold(0.98), 0; bold(0.98), 0; 0, 0; 0, 0),
    mat(0, 0; 0, 0; 0, 0; 0, 0; 0, 0; 0, bold(0.99); 0, bold(0.99)),
  );
)
$))

#pagebreak()
== Span Decoding

=== Transformation

$O -> S$

=== Source Code

- Struct: `gliner::model::output::decoded::SpanOutput`
- Transformation: `gliner::model::output::decoded::token::TensorsToDecoded`

=== Format

Same format as in span-mode.

#pagebreak()
= Post-Processing (Common)

== Span Filtering (Greedy Search)

=== Transformation

$S -> S'$

=== Source Code

- Struct: `gliner::model::output::decoded::SpanOutput`
- Transformation: `gliner::model::output::decoded::greedy::GreedySearch`

=== Format

Same as span output.

