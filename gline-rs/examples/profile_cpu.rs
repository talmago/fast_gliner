//! CPU stage profile for GLiNER2, GLiClass, and GLiFormer.
//!
//! Times host work before the session, each `session.run`, and host work after it.
//! A batch is still one session per text, which is what those runtimes do today.
//!
//! ```bash
//! cargo run --release --example profile-cpu --manifest-path gline-rs/Cargo.toml -- models
//! ```

use std::path::{Path, PathBuf};

use gliner::model::params::Parameters;
use gliner::model::profile::{GliclassProbe, Gliner2Probe, StageMillis};
use gliner::model::{
    ExtractionFieldSchema, ExtractionSchema, GLiFormer, GliformerStages, GliformerStructureStages,
    SchemaNode, StructureSchema,
};
use gliner::util::result::Result;
use orp::params::RuntimeParameters;

const WARMUP: usize = 2;
const RUNS: usize = 5;
const THREADS: usize = 4;
const BATCH: usize = 8;

fn main() -> Result<()> {
    let models = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"));
    println!("CPU intra-op threads: {THREADS}");
    println!("timed runs: {RUNS} after {WARMUP} warmup");

    let sentence = "Barack Obama met Angela Merkel in Berlin on 12 June 2014 to discuss NATO.";
    let paragraph =
        "On 12 June 2014, Barack Obama met Angela Merkel at the Chancellery in Berlin. \
        The White House said the talks covered NATO, Ukraine, and trade with the European Union. \
        Obama later flew to Paris and met François Hollande near the Eiffel Tower. \
        Mercedes-Benz sponsored the evening reception at the Adlon Hotel.";
    let ner_labels = vec![
        "person".to_string(),
        "location".to_string(),
        "organization".to_string(),
        "date".to_string(),
    ];
    let class_labels = vec![
        "politics".to_string(),
        "travel".to_string(),
        "business".to_string(),
        "science".to_string(),
    ];

    profile_gliner2(&models, sentence, paragraph, &ner_labels, &class_labels)?;
    profile_gliclass(&models, sentence, paragraph, &class_labels)?;
    profile_gliformer(&models, sentence, paragraph, &ner_labels)?;
    Ok(())
}

fn profile_gliner2(
    models: &Path,
    sentence: &str,
    paragraph: &str,
    ner_labels: &[String],
    class_labels: &[String],
) -> Result<()> {
    let dir = models.join("gliner2-multi-v1-onnx");
    println!("\n# GLiNER2 {}", dir.display());
    let probe = Gliner2Probe::open(&dir, THREADS)?;
    report_stages(
        "ner sentence",
        &repeat(|| probe.time_ner(sentence, ner_labels))?,
    );
    report_stages(
        "ner paragraph",
        &repeat(|| probe.time_ner(paragraph, ner_labels))?,
    );
    report_stages(
        "ner batch-8",
        &repeat(|| sum_stages((0..BATCH).map(|_| probe.time_ner(sentence, ner_labels))))?,
    );
    report_stages(
        "classify sentence",
        &repeat(|| probe.time_classify(sentence, class_labels))?,
    );
    let multitask_text = "NEWS REPORT: Bill Gates founded Microsoft on October 26, 2018. \
        Satya Nadella works for Microsoft.";
    let multitask_schema = ExtractionSchema::from_fields(vec![
        ExtractionFieldSchema::new(
            "document_type",
            vec!["news".into(), "report".into(), "announcement".into()],
        ),
        ExtractionFieldSchema::new("entities", vec!["person".into(), "company".into()]),
        ExtractionFieldSchema::new("founded", vec!["founded company".into()]),
        ExtractionFieldSchema::new("works_for", vec!["works for company".into()]),
        ExtractionFieldSchema::new("event_date", vec!["date".into()]),
        ExtractionFieldSchema::new("event_description", vec!["description".into()]),
    ]);
    let class_task = vec!["news".into(), "report".into(), "announcement".into()];
    report_stages(
        "extract sentence",
        &repeat(|| probe.time_extract(multitask_text, &multitask_schema))?,
    );
    report_stages(
        "multitask sentence",
        &repeat(|| {
            let extract = probe.time_extract(multitask_text, &multitask_schema)?;
            let classify = probe.time_classify(multitask_text, &class_task)?;
            Ok(add_stages(extract, classify))
        })?,
    );
    Ok(())
}

fn profile_gliclass(
    models: &Path,
    sentence: &str,
    paragraph: &str,
    labels: &[String],
) -> Result<()> {
    let dir = models.join("gliclass-small-v1.0");
    println!("\n# GLiClass {}", dir.display());
    let probe = GliclassProbe::open(&dir, THREADS)?;
    report_stages(
        "classify sentence",
        &repeat(|| probe.time_classify(sentence, labels))?,
    );
    report_stages(
        "classify paragraph",
        &repeat(|| probe.time_classify(paragraph, labels))?,
    );
    report_stages(
        "classify batch-8",
        &repeat(|| sum_stages((0..BATCH).map(|_| probe.time_classify(sentence, labels))))?,
    );
    Ok(())
}

fn profile_gliformer(
    models: &Path,
    sentence: &str,
    paragraph: &str,
    labels: &[String],
) -> Result<()> {
    let dir = models.join("gliformer-base-v1");
    println!("\n# GLiFormer {}", dir.display());
    let model = GLiFormer::from_dir(
        dir,
        Parameters::default(),
        RuntimeParameters::default().with_threads(THREADS),
    )?;
    report_gliformer(
        "ner sentence",
        &repeat_former(|| model.profile_entities(sentence, labels))?,
    );
    report_gliformer(
        "ner paragraph",
        &repeat_former(|| model.profile_entities(paragraph, labels))?,
    );
    report_gliformer(
        "ner batch-8",
        &repeat_former(|| {
            sum_former((0..BATCH).map(|_| model.profile_entities(sentence, labels)))
        })?,
    );
    let structure_text = "At Acme, Engineering includes Alice, a software engineer, and Bob, \
        a designer. Sales includes Carol, an account manager.";
    let structure_schema = StructureSchema::new([(
        "company",
        SchemaNode::object([
            ("name", SchemaNode::scalar("str")),
            (
                "departments",
                SchemaNode::array(SchemaNode::object([
                    ("name", SchemaNode::scalar("str")),
                    (
                        "employees",
                        SchemaNode::array(SchemaNode::object([
                            ("name", SchemaNode::scalar("str")),
                            ("role", SchemaNode::scalar("str")),
                        ])),
                    ),
                ])),
            ),
        ]),
    )]);
    report_structure(
        "structure sentence",
        &repeat_structure(|| model.profile_structure(structure_text, &structure_schema))?,
    );
    Ok(())
}

fn repeat(mut once: impl FnMut() -> Result<StageMillis>) -> Result<Vec<StageMillis>> {
    for _ in 0..WARMUP {
        once()?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        samples.push(once()?);
    }
    Ok(samples)
}

fn repeat_former(
    mut once: impl FnMut() -> Result<GliformerStages>,
) -> Result<Vec<GliformerStages>> {
    for _ in 0..WARMUP {
        once()?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        samples.push(once()?);
    }
    Ok(samples)
}

fn add_stages(left: StageMillis, right: StageMillis) -> StageMillis {
    StageMillis {
        pre_ms: left.pre_ms + right.pre_ms,
        onnx_ms: left.onnx_ms + right.onnx_ms,
        post_ms: left.post_ms + right.post_ms,
    }
}

fn repeat_structure(
    mut once: impl FnMut() -> Result<GliformerStructureStages>,
) -> Result<Vec<GliformerStructureStages>> {
    for _ in 0..WARMUP {
        once()?;
    }
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        samples.push(once()?);
    }
    Ok(samples)
}

fn sum_stages(parts: impl Iterator<Item = Result<StageMillis>>) -> Result<StageMillis> {
    let mut total = StageMillis::default();
    for part in parts {
        let part = part?;
        total.pre_ms += part.pre_ms;
        total.onnx_ms += part.onnx_ms;
        total.post_ms += part.post_ms;
    }
    Ok(total)
}

fn sum_former(parts: impl Iterator<Item = Result<GliformerStages>>) -> Result<GliformerStages> {
    let mut total = GliformerStages::default();
    for part in parts {
        let part = part?;
        total.pre_ms += part.pre_ms;
        total.encoder_ms += part.encoder_ms;
        total.between_ms += part.between_ms;
        total.head_ms += part.head_ms;
        total.post_ms += part.post_ms;
    }
    Ok(total)
}

fn report_stages(name: &str, samples: &[StageMillis]) {
    let pre = median(samples.iter().map(|sample| sample.pre_ms));
    let onnx = median(samples.iter().map(|sample| sample.onnx_ms));
    let post = median(samples.iter().map(|sample| sample.post_ms));
    let total = pre + onnx + post;
    println!(
        "{name:<22} pre {pre:8.2} ms ({:5.1}%)  onnx {onnx:8.2} ms ({:5.1}%)  post {post:8.2} ms ({:5.1}%)  total {total:8.2} ms",
        pct(pre, total),
        pct(onnx, total),
        pct(post, total)
    );
}

fn report_gliformer(name: &str, samples: &[GliformerStages]) {
    let pre = median(samples.iter().map(|sample| sample.pre_ms));
    let encoder = median(samples.iter().map(|sample| sample.encoder_ms));
    let between = median(samples.iter().map(|sample| sample.between_ms));
    let head = median(samples.iter().map(|sample| sample.head_ms));
    let post = median(samples.iter().map(|sample| sample.post_ms));
    let total = pre + encoder + between + head + post;
    println!(
        "{name:<22} pre {pre:8.2}  encoder {encoder:8.2} ({:4.1}%)  between {between:7.2} ({:4.1}%)  head {head:7.2} ({:4.1}%)  post {post:7.2}  total {total:8.2} ms",
        pct(encoder, total),
        pct(between, total),
        pct(head, total)
    );
}

fn report_structure(name: &str, samples: &[GliformerStructureStages]) {
    let pre = median(samples.iter().map(|sample| sample.pre_ms));
    let encoder = median(samples.iter().map(|sample| sample.encoder_ms));
    let between = median(samples.iter().map(|sample| sample.between_ms));
    let ner = median(samples.iter().map(|sample| sample.ner_head_ms));
    let structure = median(samples.iter().map(|sample| sample.structure_head_ms));
    let post = median(samples.iter().map(|sample| sample.post_ms));
    let total = pre + encoder + between + ner + structure + post;
    println!(
        "{name:<22} pre {pre:8.2}  encoder {encoder:8.2} ({:4.1}%)  ner {ner:7.2} ({:4.1}%)  structure {structure:7.2} ({:4.1}%)  post {post:7.2}  total {total:8.2} ms",
        pct(encoder, total),
        pct(ner, total),
        pct(structure, total)
    );
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap());
    values[values.len() / 2]
}

fn pct(part: f64, total: f64) -> f64 {
    if total == 0.0 {
        0.0
    } else {
        100.0 * part / total
    }
}
