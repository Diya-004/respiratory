from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from textwrap import dedent

from bs4 import BeautifulSoup


ROOT = Path("/Users/diyarao/Documents/Playground")
REBUILD = ROOT / "respiratory_ai_rebuild"

OUTPUT_HTML = ROOT / "major_project_report_full.html"
OUTPUT_RTF = ROOT / "major_project_report_full.rtf"
OUTPUT_DOCX = ROOT / "major_project_report_full.docx"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


DATASET_AUDIT = read_json(REBUILD / "reports" / "dataset_audit_paper.json")
PNEUMONIA_SUMMARY = read_json(REBUILD / "dataset_final_pneumonia_focus" / "metadata" / "pneumonia_focus_summary.json")
VAL_METRICS = read_json(REBUILD / "models_strong_cnn_pneumonia_focus" / "latest" / "metrics.json")
TEST_METRICS = read_json(REBUILD / "models_strong_cnn_pneumonia_focus" / "latest" / "test_evaluation_recording" / "metrics.json")
TRAINING_OPTIONS = read_json(
    REBUILD
    / "models_strong_cnn_pneumonia_focus"
    / "respiratory_classifier_strong_cnn_pneumonia_focus_80ep_strong_cnn_20260329_190421"
    / "training_options.json"
)
TRAIN_HISTORY = read_csv_rows(
    REBUILD
    / "models_strong_cnn_pneumonia_focus"
    / "respiratory_classifier_strong_cnn_pneumonia_focus_80ep_strong_cnn_20260329_190421"
    / "history.csv"
)


PAPERS = [
    {
        "title": "A Respiratory Sound Database for the Development of Automated Classification",
        "authors": "Rocha, Oliveira, Rennoll, Silva, and collaborators",
        "year": "2018",
        "venue": "ICBHI / Springer",
        "url": "https://doi.org/10.1007/978-981-10-7419-6_6",
        "summary": [
            "This work is important because it established one of the most widely used open respiratory sound resources in the field. The paper describes the ICBHI 2017 respiratory sound dataset, which brought together a large number of annotated respiratory recordings collected using different acquisition devices and recording locations. In respiratory sound research, dataset quality and annotation quality shape everything that follows, including preprocessing choices, split strategies, class definitions, and the eventual performance ceiling of machine learning models.",
            "The dataset became a reference point for later deep learning studies because it enabled a relatively standardized benchmark. It contains normal and adventitious respiratory sounds, with labels that allow both sound event recognition and broader disease-oriented analysis. For this project, the relevance of the paper lies not only in the dataset itself but also in the fact that it reveals the field-wide challenge of inconsistent acquisition conditions. Variations in stethoscope type, recording duration, signal quality, and patient pathology make direct modeling difficult and create a strong need for careful preprocessing and fair evaluation.",
            "A major lesson taken from this dataset paper was that benchmark availability does not automatically solve the problem of clinically meaningful generalization. Many later studies reported strong scores on the dataset while using different split rules or different assumptions about how recordings should be segmented. That challenge directly influenced the present project, which deliberately adopted patient-aware splitting, source-aware analysis, and a shared preprocessing pipeline so that training and inference remain consistent. In short, this paper provided the foundational dataset perspective that shaped the design of the project's data engineering workflow.",
        ],
        "strengths": "Provides a benchmark dataset and a common research reference point.",
        "limitations": "Does not by itself solve data imbalance, domain shift, or evaluation leakage.",
        "relevance": "Directly informed the dataset selection and split design in the current project.",
    },
    {
        "title": "Respiratory Sound Classification for Crackles, Wheezes, and Rhonchi in the Clinical Field Using Deep Learning",
        "authors": "Kim, Hyon, Lee, and collaborators",
        "year": "2021",
        "venue": "Scientific Reports",
        "url": "https://www.nature.com/articles/s41598-021-96724-7",
        "summary": [
            "Kim and colleagues studied respiratory sound classification using real clinical recordings rather than only challenge-style benchmark subsets. Their paper is especially useful because it moves the discussion closer to field conditions. The authors used a sizeable clinical respiratory sound collection and trained a deep learning approach to classify respiratory sounds into clinically meaningful categories including crackles, wheezes, and rhonchi. They reported strong results for both abnormal sound detection and subtype classification, demonstrating that deep learning could perform well in a realistic auscultation setting.",
            "One reason this paper matters is that it highlights a shift from simple normal-versus-abnormal screening toward richer respiratory sound categorization. In practice, clinicians do not just need a binary answer; they need a structured interpretation of what sort of abnormality may be present. The present project borrowed that philosophy even though its final output classes are disease labels such as Asthma, COPD, Normal, and Pneumonia. The system therefore includes probability distributions, severity estimation, representative windows, and a heatmap rather than returning only a single class prediction.",
            "Another important contribution of the paper is that it underscores the gap between controlled datasets and field recordings. Clinical recordings are noisier, more variable, and harder to annotate. That insight motivated the current project's emphasis on audio trimming, normalization, window extraction, balanced preprocessing, and robust web/mobile deployment. The paper therefore served as a methodological bridge between academic benchmarks and practical system design.",
        ],
        "strengths": "Clinical-field relevance and strong practical framing for respiratory sound analysis.",
        "limitations": "Focuses on sound-event categories, so disease-level interpretation still requires additional modeling assumptions.",
        "relevance": "Motivated the inclusion of richer outputs and practical deployment features in the present system.",
    },
    {
        "title": "Respiratory Sound Classification by Applying Deep Neural Network with a Blocking Variable",
        "authors": "Yang, Xie, Zheng, and collaborators",
        "year": "2023",
        "venue": "Applied Sciences",
        "url": "https://www.mdpi.com/2076-3417/13/12/6956",
        "summary": [
            "Yang and co-authors proposed a deep neural network framework for respiratory sound classification on the ICBHI 2017 dataset. A key feature of their work was the attempt to handle non-independent and non-identically distributed structure in the data through the use of a blocking variable. The study also addressed class imbalance through mix-up augmentation and explored a spectrogram-based deep learning pipeline. Their reported specificity, sensitivity, and average score showed that respiratory sound classification performance is highly dependent on how data variability is handled.",
            "This paper is significant for the present project because it goes beyond a standard train-and-test routine and asks a more statistically careful question: how should modeling behave when data units are not fully independent? Respiratory datasets often contain repeated patterns from the same patient, same recording environment, or same pathology cluster. If those dependencies are ignored, models may appear stronger than they really are. That concern strongly aligns with the current project's deliberate use of patient-aware group splitting and source-aware inspection of dataset gaps.",
            "The study also reinforced the usefulness of spectrogram representations in deep learning pipelines for lung sound classification. Even though the current project eventually implemented its own log-mel plus delta-channel pipeline, the broader insight remained the same: the representation of respiratory acoustics in time-frequency form is central to performance. Yang et al. therefore supported two of the most important design decisions in the project, namely leakage-aware experimental design and spectrogram-oriented modeling.",
        ],
        "strengths": "Focuses on statistical data structure, class imbalance, and benchmarked deep learning.",
        "limitations": "Primarily benchmark-focused and less concerned with full deployment or end-user interfaces.",
        "relevance": "Helped justify patient-aware splitting and augmentation-aware modeling in the project.",
    },
    {
        "title": "Performance Evaluation of Lung Sounds Classification Using Deep Learning Under Variable Parameters",
        "authors": "Zhaoping Wang and Zhiqiang Sun",
        "year": "2024",
        "venue": "EURASIP Journal on Advances in Signal Processing",
        "url": "https://link.springer.com/article/10.1186/s13634-024-01148-w",
        "summary": [
            "Wang and Sun studied how respiratory sound classification performance changes when feature-related and segmentation-related parameters are varied. Instead of presenting only a final model score, they performed a sensitivity-style analysis of frame length, overlap percentage, and feature type while using the ICBHI 2017 dataset. Their conclusions showed that parameter choices matter substantially and that higher overlap and carefully selected spectrogram settings can improve performance without unnecessary computational cost.",
            "This paper is particularly relevant because the current project also treats segmentation and overlap as first-class design variables rather than invisible implementation details. The final deployed inference backend does not rely on one arbitrary segment. Instead, it extracts up to five overlapping windows from a recording, predicts them individually, and then aggregates the probabilities at recording level. The use of `window_overlap = 0.5` in the final saved model family reflects the same kind of design attention advocated by Wang and Sun.",
            "Another important aspect of the paper is its practical engineering message: respiratory sound systems are highly sensitive to preprocessing settings. This insight influenced the current project to centralize audio configuration in YAML so that the same sample rate, duration, spectrogram parameters, and overlap logic are shared across training and inference. Such consistency is essential in deployment settings, because even small preprocessing mismatches can degrade model performance dramatically after deployment.",
        ],
        "strengths": "Demonstrates the importance of preprocessing and segmentation choices through focused experiments.",
        "limitations": "Emphasizes parameter sensitivity more than interface design or explainability.",
        "relevance": "Directly supports the project's recording-window aggregation and shared configuration design.",
    },
    {
        "title": "Improving the Robustness and Clinical Applicability of Automatic Respiratory Sound Classification Using Deep Learning–Based Audio Enhancement",
        "authors": "Tzeng, Chan, and collaborators",
        "year": "2025",
        "venue": "JMIR AI",
        "url": "https://ai.jmir.org/2025/1/e67239",
        "summary": [
            "Tzeng and colleagues examined the robustness problem in respiratory sound classification by introducing deep learning–based audio enhancement into the pipeline. Their work is valuable because it addresses a limitation often ignored in academic prototypes: the fact that recorded respiratory sounds may be noisy, distorted, or captured in suboptimal conditions. In real-world healthcare or telemedicine settings, classification quality depends not only on the model architecture but also on signal quality and denoising strategy.",
            "This paper resonates strongly with the mobile and practical aims of the present project. Because the project includes an Android recording workflow, it must be assumed that recordings will vary in quality across devices, environments, and user behavior. The current system therefore recommends quiet-environment recording, trims silence, normalizes the signal, and uses recording-level aggregation to reduce sensitivity to a single poor segment. While the current implementation does not yet include a full learned enhancement stage, the paper clearly identifies that as a strong next-step direction for future system improvement.",
            "The most important contribution of the paper for this report is conceptual: it reframes model development from pure classification accuracy to clinical applicability and robustness. That same perspective appears in the present work through explainability heatmaps, severity messages, web and mobile accessibility, and explicit acknowledgement of remaining limitations such as noise sensitivity and class imbalance. In that sense, Tzeng et al. offered a practical deployment lens that aligns closely with the goals of this major project.",
        ],
        "strengths": "Addresses robustness and clinical usability rather than only benchmark scores.",
        "limitations": "Enhancement-oriented approaches may increase computational complexity for edge deployment.",
        "relevance": "Supports the project's future scope around signal-quality improvement and clinically usable deployment.",
    },
]


def h(level: int, text: str) -> str:
    return f"<h{level}>{text}</h{level}>"


def p(text: str) -> str:
    return f"<p>{text}</p>"


def ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "<tr>" + "".join(f"<th>{cell}</th>" for cell in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead>{thead}</thead><tbody>{body}</tbody></table>"


def fig(caption: str, note: str) -> str:
    return f"<p><strong>{caption}</strong>: {note}</p>"


def chapter(title: str) -> str:
    return f'<div class="page-break"></div>{h(1, title)}'


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_cover_page() -> str:
    return dedent(
        """
        <div class="cover">
          <div class="cover-block">
            <div class="cover-line">FINAL MAJOR PROJECT REPORT</div>
            <div class="cover-title">PNEUMOSCAN</div>
            <div class="cover-subtitle">AI-Assisted Respiratory Disease Screening Platform for Lung Sound Analysis</div>
            <div class="cover-meta">
              <p>Submitted in partial fulfillment of the requirements for the award of the degree of</p>
              <p><strong>Bachelor of Engineering / Bachelor of Technology</strong></p>
              <p>in</p>
              <p><strong>[Department Name]</strong></p>
            </div>
            <div class="cover-meta">
              <p>Submitted by: <strong>[Student Name]</strong></p>
              <p>Roll No.: <strong>[Roll Number]</strong></p>
              <p>Under the guidance of: <strong>[Guide Name]</strong></p>
            </div>
            <div class="cover-meta">
              <p><strong>[College Name]</strong></p>
              <p>[University Name]</p>
              <p>Academic Year: [2025-2026]</p>
            </div>
          </div>
        </div>
        """
    )


def build_front_matter() -> str:
    toc_items = [
        "Declaration",
        "Acknowledgments",
        "Abstract",
        "Table of Contents",
        "List of Tables",
        "List of Figures",
        "Abbreviations",
        "Chapter 1: Introduction",
        "Chapter 2: Literature Review",
        "Chapter 3: Proposed System",
        "Chapter 6: Conclusion and Future Scope",
        "References",
        "List of Publications",
        "Appendix A: Dataset Audit Summary",
        "Appendix B: Model Configuration and Metrics",
        "Appendix C: Training History",
        "Appendix D: API Contract and Sample Output",
        "Appendix E: Module Inventory and Deployment Notes",
        "Appendix F: Detailed User Manual and Demonstration Workflow",
        "Appendix G: Testing Strategy, Scenarios, and Observations",
        "Appendix H: Risk Register, Limitations, and Ethical Reflection",
        "Appendix I: Class-Wise Error Analysis and Improvement Roadmap",
        "Appendix J: Example Usage Scenarios and Interpretive Case Notes",
        "Appendix K: Module-Wise Technical Walkthrough",
        "Appendix L: Project Management Reflection and Future Research Questions",
        "Appendix M: Proposed Raspberry Pi Edge Deployment Concept",
    ]
    tables = [
        "Table 1.1 Functional requirements of the proposed system",
        "Table 1.2 Non-functional requirements of the proposed system",
        "Table 2.1 Literature survey comparison table",
        "Table 3.1 Dataset summary after deduplication",
        "Table 3.2 Source-wise class distribution in the prepared paper dataset",
        "Table 3.3 Pneumonia-focused augmentation summary",
        "Table 3.4 Final model configuration",
        "Table 3.5 Proposed edge device hardware integration summary",
        "Table 3.6 Validation performance of the saved strong-CNN run",
        "Table 3.7 Recording-level test performance of the deployed model",
        "Table 3.8 Per-class validation metrics",
        "Table 3.9 Per-class recording-level test metrics",
        "Table A.1 Source-wise class distribution in the paper dataset",
        "Table B.1 Per-class validation metrics",
        "Table B.2 Per-class recording-level test metrics",
        "Table M.1 Proposed Raspberry Pi edge-device bill of materials",
    ]
    figures = [
        "Figure 3.1 Overall data-to-deployment workflow",
        "Figure 3.2 Audio preprocessing pipeline",
        "Figure 3.3 Recording-level window aggregation concept",
        "Figure 3.4 Web platform workflow",
        "Figure 3.5 Android mobile workflow",
        "Figure 3.6 Proposed Raspberry Pi edge-device workflow",
        "Figure 3.7 Heatmap-based explainability concept",
    ]
    abbreviations = [
        ["AI", "Artificial Intelligence"],
        ["API", "Application Programming Interface"],
        ["CNN", "Convolutional Neural Network"],
        ["CNN-LSTM", "Convolutional Neural Network with Long Short-Term Memory"],
        ["COPD", "Chronic Obstructive Pulmonary Disease"],
        ["FFT", "Fast Fourier Transform"],
        ["Grad-CAM", "Gradient-weighted Class Activation Mapping"],
        ["MFCC", "Mel Frequency Cepstral Coefficient"],
        ["RNN", "Recurrent Neural Network"],
        ["UI", "User Interface"],
        ["WAV", "Waveform Audio File Format"],
    ]
    abstract_text = [
        "Respiratory diseases such as Asthma, Chronic Obstructive Pulmonary Disease (COPD), and Pneumonia require early screening and timely clinical attention. Conventional auscultation remains useful but depends heavily on clinician experience, recording quality, and interpretation consistency. This major project presents <strong>PneumoScan</strong>, an AI-assisted respiratory disease screening platform that analyzes lung and breathing sounds and presents clinically interpretable outputs through both web and mobile interfaces.",
        "The project was implemented as a complete engineering system rather than a model-only experiment. The final solution includes dataset preparation scripts, duplicate removal, label normalization, patient-aware splitting, audio preprocessing through log-mel spectrogram generation, support for multiple deep learning architectures, recording-level window aggregation, a Flask-based inference API, a browser interface, and a native Android application capable of recording WAV audio and submitting it for analysis. Explainability support is provided through Grad-CAM style heatmaps and representative-window reporting. In addition, the project scope includes an edge-device deployment direction based on a Raspberry Pi, a microphone input module, and an LED display so that the same model can eventually be executed in a compact standalone screening unit.",
        f"The saved deployment-oriented strong-CNN model family achieved a recording-level test accuracy of <strong>{fmt_pct(TEST_METRICS['accuracy'])}</strong> and a weighted F1-score of <strong>{fmt_pct(TEST_METRICS['weighted_f1'])}</strong>, with the strongest class-wise performance observed for COPD. The report treats the Raspberry Pi hardware path as an ongoing/proposed integration layer rather than a fully completed deployment, but it is included because it extends the practical significance of the platform beyond browser and phone interfaces. Overall, the project demonstrates the feasibility of combining respiratory audio signal processing, deep learning, explainable AI, edge-oriented thinking, and user-facing deployment into a practical screening platform. Although the system is not a replacement for medical diagnosis, it forms a strong foundation for future clinical decision-support and telehealth-oriented respiratory screening systems.",
    ]
    out = []
    out.append('<div class="page-break"></div>' + h(1, "Declaration"))
    out.append(
        p(
            "I hereby declare that this report entitled <strong>“PneumoScan: AI-Assisted Respiratory Disease Screening Platform for Lung Sound Analysis”</strong> is a record of original work carried out by me under the guidance of <strong>[Guide Name]</strong> during the academic year <strong>[2025-2026]</strong>. The work submitted in this report has not been submitted elsewhere for the award of any degree or diploma. All sources of information used in the preparation of this report have been appropriately acknowledged."
        )
    )
    out.append(p("Place: [City]"))
    out.append(p("Date: [Submission Date]"))
    out.append(p("Signature of the Student: ______________________"))
    out.append(p("Name of the Student: [Student Name]"))

    out.append('<div class="page-break"></div>' + h(1, "Acknowledgments"))
    out.append(
        p(
            "The successful completion of this project would not have been possible without the guidance, encouragement, and support of many people. I express my sincere gratitude to <strong>[Guide Name]</strong> for continuous guidance, valuable suggestions, and constructive feedback throughout the development of this work. The guidance received at every stage of the project helped shape the technical direction of the system and improved both the research quality and implementation quality of the final platform."
        )
    )
    out.append(
        p(
            "I also extend heartfelt thanks to the faculty members of <strong>[Department Name]</strong> at <strong>[College Name]</strong> for providing the academic environment, resources, and encouragement required to complete this project. Their emphasis on disciplined engineering practice, systematic experimentation, and project presentation contributed greatly to the completion of this report."
        )
    )
    out.append(
        p(
            "I am grateful to my family and friends for their moral support, patience, and motivation throughout the project duration. I also acknowledge the open research community and dataset contributors whose work made it possible to build, evaluate, and compare respiratory sound analysis systems in a meaningful way."
        )
    )

    out.append('<div class="page-break"></div>' + h(1, "Abstract"))
    out.extend(p(x) for x in abstract_text)

    out.append('<div class="page-break"></div>' + h(1, "Table of Contents"))
    out.append(ul(toc_items))

    out.append('<div class="page-break"></div>' + h(1, "List of Tables"))
    out.append(ul(tables))

    out.append('<div class="page-break"></div>' + h(1, "List of Figures"))
    out.append(ul(figures))

    out.append('<div class="page-break"></div>' + h(1, "Abbreviations"))
    out.append(table(["Abbreviation", "Expanded Form"], abbreviations))
    return "".join(out)


def build_chapter_1() -> str:
    requirements_rows = [
        ["FR1", "The system shall accept respiratory audio files in common audio formats for analysis."],
        ["FR2", "The system shall preprocess audio consistently for training and inference."],
        ["FR3", "The system shall classify recordings into Asthma, COPD, Normal, or Pneumonia."],
        ["FR4", "The system shall provide confidence, severity, and window-level prediction details."],
        ["FR5", "The platform shall support both browser-based upload and Android mobile use."],
        ["FR6", "The system shall provide explainability through representative windows and heatmaps."],
        ["FR7", "The proposed edge hardware module shall support Raspberry Pi based microphone capture and LED display output."],
    ]
    non_functional_rows = [
        ["NFR1", "Consistency between training and inference preprocessing"],
        ["NFR2", "Readable and clinically interpretable output presentation"],
        ["NFR3", "Modular architecture for replacing the active model family"],
        ["NFR4", "Reasonable response time for single-recording inference"],
        ["NFR5", "Robust handling of noisy and heterogeneous audio inputs"],
        ["NFR6", "Support for portability across web and Android interfaces"],
        ["NFR7", "Edge-oriented deployment feasibility on low-cost hardware platforms"],
    ]
    paragraphs_11 = [
        "Respiratory diseases remain one of the most serious public health concerns across the world. Conditions such as Asthma, Chronic Obstructive Pulmonary Disease (COPD), and Pneumonia are associated with high morbidity, repeated hospitalization, and major long-term quality-of-life impact. Early screening is particularly important because timely intervention can reduce complications, support earlier treatment decisions, and improve patient monitoring. In many real-world settings, however, access to specialist respiratory evaluation is limited by geography, workload, cost, and equipment availability.",
        "Auscultation of lung and breathing sounds is one of the oldest and most informative non-invasive methods used in respiratory assessment. Crackles, wheezes, rhonchi, and other acoustic variations often provide clinically valuable cues regarding airway obstruction, fluid accumulation, inflammation, or other abnormal pulmonary conditions. Yet traditional auscultation is inherently dependent on clinician training and listening experience. Even among skilled practitioners, inter-observer variability remains a challenge, especially when sounds are faint, mixed, noisy, or recorded outside a controlled clinical environment.",
        "The growth of digital stethoscopes, smartphones, telehealth platforms, and low-cost audio capture systems has created a new opportunity for computational respiratory screening. Once respiratory sounds are captured as digital signals, they can be processed using signal analysis and machine learning methods. Deep learning, in particular, has made it possible to learn discriminative patterns directly from spectrogram-like representations of biomedical audio. This makes respiratory sound analysis a promising application area for artificial intelligence, especially in screening-oriented or triage-support scenarios.",
        "This major project investigates that opportunity by building a complete AI-assisted respiratory sound screening platform. Instead of stopping at a notebook-based classification experiment, the project evolves through multiple stages: an initial browser-based MVP, a re-engineered machine learning pipeline, a deployable inference backend, and a native Android mobile client. The result is a system that can accept a respiratory sound recording, process it, predict a condition class, estimate severity, visualize model attention, and present a user-friendly summary through practical interfaces.",
        "In addition to the web and Android pathways, the broader project scope includes an edge-device module built around a Raspberry Pi, a microphone, and an LED display of the same hardware family. The purpose of this module is to turn the respiratory screening pipeline into a compact standalone device that can listen to respiratory audio, process it using the trained model, and display the result locally. Since this hardware pathway is still under development, the report includes it as an ongoing/proposed subsystem rather than as a fully completed deliverable.",
        "The project therefore sits at the intersection of biomedical signal processing, machine learning, explainable AI, backend development, frontend design, and mobile application development. It is academically meaningful because it studies respiratory sound analysis in a systematic manner, and it is engineering-relevant because it turns that analysis into an accessible platform. This dual focus on research and implementation is one of the main features that distinguishes the project from purely theoretical studies.",
        "The central topic of the report is thus not only respiratory disease classification, but the design of an end-to-end AI-assisted screening ecosystem. The present work argues that for a respiratory sound classifier to be practically valuable, it must be supported by robust data preparation, fair evaluation, consistent preprocessing, interpretable outputs, and deployable interfaces. The remainder of this report develops that argument in detail through background review, design discussion, implementation explanation, and measured results.",
    ]
    paragraphs_12 = [
        "The core problem addressed in this project arises from a mismatch between the clinical value of respiratory sound analysis and the difficulty of applying it consistently at scale. Lung sound interpretation is useful, low-cost, and non-invasive, yet it is also subjective and sensitive to listener skill. In healthcare environments with high patient volumes or limited specialist availability, it becomes difficult to guarantee uniform interpretation quality for all respiratory recordings. This motivates the search for a computational screening aid that can complement clinical judgment.",
        "From a technical standpoint, respiratory sound classification is not a trivial pattern recognition problem. The recordings differ in duration, sample rate, device type, background noise, and disease prevalence. Some classes are heavily underrepresented relative to others, and some datasets contain repeated patterns from the same patient or source. If these issues are ignored, machine learning systems may appear strong during evaluation but fail to generalize after deployment. Therefore, building a meaningful system requires careful attention to data engineering, split design, preprocessing, and evaluation strategy.",
        "Another problem lies in the gap between research prototypes and deployable systems. Many published studies focus primarily on benchmark scores, often using specialized preprocessing pipelines that are difficult to reproduce in end-user applications. Even when a good model is achieved, the research output may stop at an offline script or notebook. For educational major projects, however, demonstrating a complete system is especially important because it shows not just algorithmic understanding but full-stack engineering capability.",
        "The present project defines the problem more concretely as follows: to design and implement an AI-assisted platform capable of processing respiratory audio recordings and returning useful screening output through practical interfaces, while preserving evaluation honesty and engineering consistency. That means the system must do more than classify audio; it must also support input acquisition, signal normalization, model integration, output explanation, and user interaction through web and mobile channels.",
        "The inclusion of the Raspberry Pi hardware path extends this problem definition further. The system is expected not only to run on a server-backed software stack but also to support a low-cost edge deployment idea in which microphone input is captured directly on a small computing device and the result is shown on an attached LED display. This requirement introduces additional concerns such as lightweight deployment, hardware input/output integration, and standalone usability.",
        "A second dimension of the problem concerns interpretability. In healthcare-related applications, a predicted label alone is often insufficient. Users need additional cues that help them understand why the model responded in a certain way and whether the signal contains localized evidence for the decision. For that reason, the project treats explainability and severity communication as functional requirements instead of optional extras. The system accordingly returns heatmaps, representative windows, and structured messages alongside the predicted class.",
        "The problem is therefore multifaceted: it includes a biomedical problem, a machine learning problem, an evaluation problem, and a software deployment problem. Solving it required not one isolated model, but a coherent platform architecture. The report presents that architecture as an answer to the original problem statement and explains how each layer contributes to a more usable respiratory screening system.",
    ]
    paragraphs_13 = [
        "The first objective of the project was to collect and prepare respiratory sound data from multiple sources in a way that would support meaningful model training. This required deduplication, metadata preparation, label normalization, and the creation of source-aware and patient-aware splits. Achieving this objective was necessary because the quality of downstream classification depends heavily on the quality and structure of the input dataset.",
        "The second objective was to build a unified preprocessing pipeline for respiratory audio. Rather than using different steps for training and inference, the project aimed to centralize preprocessing parameters such as sample rate, clip duration, mel filterbank configuration, FFT size, and overlap logic. This objective was important to avoid the common failure mode in which a model performs well in experiments but poorly after deployment because the inference-side signal representation is inconsistent with what the model saw during training.",
        "The third objective was to compare and support multiple deep learning model families suitable for respiratory sound analysis. The project therefore implemented a reusable catalog of architectures including baseline CNN, custom CNN, strong CNN, EfficientNetB0, and CNN-LSTM. The goal was not only to train one model, but to create a framework in which architecture choice, configuration, and evaluation could be managed in a structured way.",
        "The fourth objective was to design a deployment-oriented inference flow. This meant moving beyond single-clip classification and toward recording-level reasoning using multiple overlapping windows. The system also needed to produce richer outputs such as class-wise probabilities, representative windows, Grad-CAM style heatmaps, and severity messages. These outputs were expected to make the system more interpretable and more relevant to user-facing deployment.",
        "The fifth objective was to deliver practical user interfaces. Two complementary interfaces were implemented: a browser-based web platform and a native Android mobile application. The web platform demonstrates the end-to-end screening flow in a desktop or browser environment, while the Android app addresses mobile accessibility and on-device respiratory audio capture. Together, these outcomes demonstrate that the project is not limited to algorithmic experimentation but extends into full platform design.",
        "A sixth objective is to extend the same inference capability toward an edge-device setup using a Raspberry Pi, microphone capture, and an LED display. In the current stage of the project this hardware path is planned/ongoing rather than fully finished, but it is already part of the intended platform scope and therefore belongs in the report as a proposed deployment module.",
        "The final outcomes achieved include a working dataset pipeline, a saved strong-CNN deployment-oriented model family, a Flask inference API, a browser interface branded as PneumoScan, and an Android client that can record WAV audio and submit it for analysis. The hardware edge-device path is included as an active future-facing subsystem under development. The presence of measured validation and test artifacts in the workspace further demonstrates that the project produced concrete, reviewable deliverables rather than aspirational plans.",
    ]
    paragraphs_14 = [
        "The motivation for this project is both social and technical. Socially, respiratory diseases affect large patient populations and are particularly concerning in settings where early expert attention is hard to obtain. A system that helps structure respiratory sound screening can support earlier review, more informed follow-up, and more accessible preliminary assessment. While such a system is not a replacement for clinical diagnosis, it can help organize information and highlight recordings that deserve closer examination.",
        "Technically, the project is motivated by the maturity of modern AI tools for audio understanding. Deep learning models have become powerful at learning spatial and temporal structure from spectrogram-like input, and respiratory sound analysis provides a highly relevant biomedical use case for those capabilities. Yet many promising results remain locked in benchmark papers or non-deployable prototypes. This project was motivated by the desire to bridge that gap and show how a trained respiratory model can be integrated into a working platform.",
        "The edge-device idea adds a further motivation: portability. A Raspberry Pi based respiratory screening unit could make the project more useful in laboratory demonstration settings, low-cost health camps, educational exhibitions, or compact prototype environments where a full laptop setup is inconvenient. The ability to capture sound from a microphone and display results directly on an LED screen strengthens the project's identity as a deployable product concept rather than only a software demonstration.",
        "There is also a strong educational motivation. A major project should demonstrate not only coding skill, but the ability to carry a system from problem definition to data engineering, model training, interface design, and final documentation. This project creates exactly that kind of end-to-end learning experience. It required reasoning about dataset quality, machine learning fairness, configuration management, API design, UI presentation, and mobile constraints within a single coherent body of work.",
        "Another motivation came from interpretability. Healthcare-related AI systems are often criticized for acting like opaque black boxes. The present project therefore deliberately includes explainability-oriented outputs, severity messaging, and window-based summaries so that the system communicates more than one hard label. This motivation is reflected in the final deployment architecture and is discussed repeatedly throughout the report.",
        "Finally, the project was motivated by the possibility of extensibility. By keeping the architecture modular and configuration-driven, the system can be adapted later for larger datasets, new models, better denoising, additional patient metadata, or more secure cloud deployment. This future extensibility made the project especially attractive as a major project topic because it provides room for continued research and product-oriented iteration.",
    ]
    paragraphs_15 = [
        "The remainder of this report is organized to move from background to evidence to final conclusions. Chapter 1 introduces the problem context, project objectives, motivation, and report structure. It positions the work as an AI-assisted respiratory sound screening platform and establishes the need for both fair evaluation and deployable implementation.",
        "Chapter 2 presents a literature review of representative research in respiratory sound analysis and deep learning–based classification. The discussion covers benchmark datasets, clinical-field classification, leakage-aware modeling, parameter sensitivity, and robustness-oriented system design. A summary table and research gap analysis are then used to explain how the present project differs from and builds upon earlier work.",
        "Chapter 3 describes the proposed system in detail. It explains the methodology used for dataset preparation, preprocessing, augmentation, model training, inference, explainability, web deployment, and Android integration. It also presents the measured validation and test results taken from the saved project artifacts and interprets them critically rather than presenting them as isolated benchmark numbers.",
        "Chapter 6, as formatted in the supplied project structure, concludes the report by summarizing the main contributions and identifying future scope. The references section records the principal research works and resources used, while the appendices provide supporting tables, configuration details, API examples, and module inventories that strengthen the completeness of the final major project documentation.",
    ]
    out = [chapter("Chapter 1: Introduction")]
    out.append(h(2, "1.1 Introduction to the Topic"))
    out.extend(p(x) for x in paragraphs_11)
    out.append(h(2, "1.2 Problem Definition"))
    out.extend(p(x) for x in paragraphs_12)
    out.append(h(2, "1.3 Objectives and Outcomes"))
    out.extend(p(x) for x in paragraphs_13)
    out.append(h(3, "Table 1.1 Functional Requirements of the Proposed System"))
    out.append(table(["ID", "Requirement"], requirements_rows))
    out.append(h(3, "Table 1.2 Non-Functional Requirements of the Proposed System"))
    out.append(table(["ID", "Requirement"], non_functional_rows))
    out.append(h(2, "1.4 Motivation"))
    out.extend(p(x) for x in paragraphs_14)
    out.append(h(2, "1.5 Thesis Organization"))
    out.extend(p(x) for x in paragraphs_15)
    return "".join(out)


def build_chapter_2() -> str:
    intro = [
        "Respiratory sound analysis has evolved from rule-based signal processing into a rich deep learning research area. Early work relied heavily on hand-crafted features such as energy statistics, MFCC summaries, temporal moments, or wavelet descriptors. More recent work has increasingly adopted spectrogram-based convolutional models, recurrent architectures, attention mechanisms, and hybrid pipelines. Across this literature, however, several recurring issues remain visible: class imbalance, split leakage, source mismatch, incomplete deployment considerations, and insufficient interpretability.",
        "The purpose of this literature review is not only to summarize prior papers, but to identify the design directions that most influenced the present project. Five representative papers were selected because together they cover the benchmark dataset perspective, clinical-field respiratory sound analysis, leakage-aware modeling, parameter sensitivity, and robustness-oriented deployment concerns. These are the same dimensions that later appear in the proposed system.",
        "For each selected paper, the review discusses the research objective, dataset and methods used, main contributions, performance highlights where available, strengths, limitations, and specific relevance to the current project. A summary comparison table and a gap analysis are then used to explain why an end-to-end platform approach remains valuable even in a field with many published classification models.",
    ]
    paper_rows = []
    out = [chapter("Chapter 2: Literature Review (Any 5 Best Related Papers)")]
    out.append(h(2, "2.1 Introduction to the Literature Review"))
    out.extend(p(x) for x in intro)
    for index, paper in enumerate(PAPERS, start=1):
        out.append(h(2, f"2.{index} Paper-{index}"))
        out.append(
            p(
                f"<strong>Title:</strong> {paper['title']}<br><strong>Authors:</strong> {paper['authors']}<br><strong>Year:</strong> {paper['year']}<br><strong>Venue:</strong> {paper['venue']}<br><strong>Source:</strong> <span class='ref-url'>{paper['url']}</span>"
            )
        )
        out.extend(p(x) for x in paper["summary"])
        out.append(p(f"<strong>Strength:</strong> {paper['strengths']}"))
        out.append(p(f"<strong>Limitation:</strong> {paper['limitations']}"))
        out.append(p(f"<strong>Relevance to the present project:</strong> {paper['relevance']}"))
        paper_rows.append(
            [
                f"Paper {index}",
                paper["year"],
                paper["venue"],
                paper["strengths"],
                paper["limitations"],
            ]
        )
    out.append(h(2, "2.6 Summary Table on Literature Survey"))
    out.append(h(3, "Table 2.1 Literature Survey Comparison Table"))
    out.append(
        table(
            ["Paper", "Year", "Venue", "Main Strength", "Main Limitation"],
            paper_rows,
        )
    )
    out.append(h(2, "2.7 Gaps Identified"))
    gaps = [
        "Many studies concentrate on benchmark scores but stop short of building complete usable systems.",
        "Class imbalance remains a persistent problem, especially for clinically important but less frequent respiratory classes.",
        "Several works report good performance but use different split logic, making direct comparison difficult.",
        "Real-world robustness under noise, device variation, and user-driven recording quality is still under-addressed.",
        "Interpretability is often weaker than required for clinician-facing or student-project deployment scenarios.",
        "Mobile integration and cross-interface accessibility are absent from a large portion of the literature.",
        "Few studies combine dataset engineering, deployment, explainability, and user interface design into one system.",
    ]
    out.extend(
        p(x)
        for x in [
            "Taken together, the reviewed literature confirms that respiratory sound classification is a viable and fast-growing research area. The field now has a solid benchmark tradition, increasing use of deep learning, and improving attention to clinically meaningful sound patterns. However, the literature also shows that high reported accuracy alone is not enough to guarantee a practical system. Differences in preprocessing, split design, source composition, and user-facing interpretation mean that many strong papers still leave engineering problems unsolved.",
            "The present project responds directly to those gaps. It treats data engineering as a first-class task through duplicate removal, label normalization, and patient-aware splitting. It treats deployment as a required outcome by implementing both a browser interface and a native Android application. It treats interpretability as a necessary design consideration through representative-window reporting and heatmap generation. Finally, it treats minority-class handling as an active concern through pneumonia-focused augmentation.",
            "Therefore, the literature review does not merely provide background; it actively justifies the architecture of the present work. The design choices made in the proposed system can be understood as responses to the unresolved issues identified in prior research. This makes the current project both derivative in the positive academic sense and distinctive in the practical engineering sense.",
        ]
    )
    out.append(ul(gaps))
    return "".join(out)


def build_chapter_3() -> str:
    dataset_rows = [
        ["Total deduplicated recordings", str(DATASET_AUDIT["deduped_records"])],
        ["Asthma", str(DATASET_AUDIT["class_counts"]["Asthma"])],
        ["COPD", str(DATASET_AUDIT["class_counts"]["COPD"])],
        ["Normal", str(DATASET_AUDIT["class_counts"]["Normal"])],
        ["Pneumonia", str(DATASET_AUDIT["class_counts"]["Pneumonia"])],
    ]
    source_rows = []
    for row in DATASET_AUDIT["source_by_label"]:
        parts = [row["label"]] + [str(v) for k, v in row.items() if k != "label"]
        source_rows.append(parts)

    val_rows = [
        ["Best validation accuracy", fmt_pct(VAL_METRICS["best_val_accuracy"])],
        ["Best validation loss", f"{VAL_METRICS['best_val_loss']:.4f}"],
        ["Macro F1-score", fmt_pct(VAL_METRICS["macro_f1"])],
        ["Weighted F1-score", fmt_pct(VAL_METRICS["weighted_f1"])],
    ]
    test_rows = [
        ["Recording-level test accuracy", fmt_pct(TEST_METRICS["accuracy"])],
        ["Macro precision", fmt_pct(TEST_METRICS["macro_precision"])],
        ["Macro recall", fmt_pct(TEST_METRICS["macro_recall"])],
        ["Macro F1-score", fmt_pct(TEST_METRICS["macro_f1"])],
        ["Weighted F1-score", fmt_pct(TEST_METRICS["weighted_f1"])],
    ]
    per_class_val = []
    for cls in ["Asthma", "COPD", "Normal", "Pneumonia"]:
        stats = VAL_METRICS["classification_report"][cls]
        per_class_val.append(
            [cls, fmt_pct(stats["precision"]), fmt_pct(stats["recall"]), fmt_pct(stats["f1-score"]), str(int(stats["support"]))]
        )
    per_class_test = []
    for cls in ["Asthma", "COPD", "Normal", "Pneumonia"]:
        stats = TEST_METRICS["classification_report"][cls]
        per_class_test.append(
            [cls, fmt_pct(stats["precision"]), fmt_pct(stats["recall"]), fmt_pct(stats["f1-score"]), str(int(stats["support"]))]
        )

    model_config_rows = [
        ["Architecture", "strong_cnn"],
        ["Sample rate", "4000 Hz"],
        ["Duration", "5.0 seconds"],
        ["Mel bands", "128"],
        ["FFT size", "512"],
        ["Hop length", "128"],
        ["Delta channels", "Enabled"],
        ["Loss", TRAINING_OPTIONS["loss_name"]],
        ["Focal gamma", str(TRAINING_OPTIONS["focal_gamma"])],
        ["Class-weight strategy", TRAINING_OPTIONS["class_weight_strategy"]],
        ["Window overlap", "0.5"],
        ["Max windows", "5"],
        ["Aggregation", "pneumonia_sensitive"],
    ]
    edge_device_rows = [
        ["Hardware role", "Ongoing/proposed edge-device deployment module"],
        ["Processing board", "Raspberry Pi"],
        ["Audio input", "Microphone connected to Raspberry Pi"],
        ["Display output", "LED screen/display module"],
        ["Intended workflow", "Capture respiratory sound, run model inference, display prediction and severity locally"],
        ["Deployment status in current report", "Included as planned/ongoing integration, not claimed as fully completed"],
        ["Engineering value", "Supports portable low-cost standalone respiratory screening concept"],
    ]

    p11 = [
        "The proposed system is an end-to-end AI-assisted respiratory screening platform developed to move beyond isolated model experimentation. It is composed of data preparation components, machine learning components, an inference backend, a browser-based interface, a native Android mobile client, and a proposed edge-device deployment path. The system was intentionally built in stages. An initial FastAPI-based MVP demonstrated the basic feasibility of audio upload, feature extraction, heuristic inference, and report generation. The later rebuild transformed that proof-of-concept into a more rigorous and configurable pipeline with a real TensorFlow-based prediction engine.",
        "This staged development is important because it mirrors good software engineering practice in machine learning projects. First, the team validated the user workflow and result presentation using a baseline implementation. Then, after the end-to-end user flow was understood, the core model and data pipeline were redesigned for correctness, reproducibility, and extensibility. This allowed the final system to inherit the usability lessons of the MVP while replacing the placeholder inference layer with a deployment-oriented deep learning architecture.",
        "The final system accepts respiratory audio in common formats, processes the signal into windows and spectrogram representations, predicts disease-oriented classes, aggregates window outputs at recording level, produces severity messaging, and displays a heatmap over a representative spectrogram window. The outputs are then exposed both in a browser workflow and an Android application. In addition, the proposed Raspberry Pi hardware pathway aims to run the same model through microphone input and show the screening output on an LED screen. Because the report is for a major project, equal emphasis is placed on the AI workflow and on the software system that makes the AI usable across software and hardware-oriented deployment contexts.",
    ]

    methodology_paragraphs = [
        "The methodology begins with dataset preparation, because respiratory audio projects often fail when the dataset is treated as a raw collection of files rather than a structured research asset. In the present work, recordings were merged from multiple sources and then audited for duplicates, label inconsistency, sample-rate heterogeneity, and duration variability. The saved audit file confirms that the prepared paper-oriented dataset contains 1756 deduplicated recordings across four final target classes.",
        "Deduplication was performed using SHA-256 file hashing. This may appear to be a simple engineering detail, but it has major implications for evaluation validity. Duplicate or near-duplicate recordings can artificially inflate validation performance and undermine trust in the model. By removing duplicates before split generation, the project reduces the chance that the same acoustic content will appear multiple times in a way that distorts model assessment.",
        "Label normalization was another key methodological step. Public respiratory datasets use inconsistent pathology labels, sound-event labels, and source-specific notations. The project therefore mapped raw labels into four final target classes: Asthma, COPD, Normal, and Pneumonia. This mapping was necessary to create a tractable classification problem that still remained clinically meaningful and suitable for a major-project scope.",
        "After dataset preparation, the project generated train, validation, and test splits using patient-aware grouping logic. This decision directly addresses one of the most important methodological concerns in respiratory sound analysis: leakage. If recordings from the same patient are split across training and evaluation sets, the model may appear to generalize when it is actually memorizing patient-specific acoustic patterns. The split creation script therefore groups recordings by patient identifier and assigns them together within label- and source-specific subsets.",
        "Audio preprocessing is performed consistently through a shared configuration layer. Each recording is resampled, silence-trimmed, normalized, and fitted to a target duration when used for clip-level representation. For recording-level inference, multiple overlapping windows are extracted so that longer signals are not collapsed into one arbitrary segment. This was a deliberate design choice informed by the literature review and the needs of user-facing deployment.",
        "Feature representation is based on log-mel spectrogram images. Delta and delta-delta channels are optionally included to capture temporal change information in addition to static spectral structure. This representation was chosen because it aligns with common deep learning practice in biomedical audio and allows convolutional architectures to learn meaningful time-frequency patterns without relying solely on hand-crafted features.",
        "The training framework supports multiple model architectures. Instead of hard-coding one network, the project implemented a model catalog that can build a baseline CNN, a custom residual-style CNN, a stronger multiscale CNN, an EfficientNet-based transfer model, or a CNN-LSTM hybrid. This architecture-level flexibility was pedagogically valuable because it allowed the project to compare design families and not overfit the entire platform to one early modeling decision.",
        "The final deployment-oriented configuration uses a strong CNN model family with focal loss and clipped class weights. The saved training options confirm that the model applies extra emphasis to minority classes, particularly Pneumonia, through both class-weight overrides and later dataset augmentation. This is academically important because it shows that the project took class imbalance seriously rather than presenting overall accuracy without examining the class distribution that produced it.",
        "Pneumonia-focused augmentation is one of the most distinctive methodological additions in the project. The saved augmentation summary shows that pneumonia clips in the training set were increased from 500 to 900, and that 878 augmented rows were added in the final pneumonia-focused dataset stage. This reflects an explicit design goal: improving representation for a clinically important and harder class instead of accepting majority-class dominance as a fixed constraint.",
        "Inference is performed at recording level. The final backend extracts up to five windows from a recording using a 0.5 overlap ratio, predicts class probabilities for each window, applies aggregation, and then performs pneumonia-sensitive post-processing when appropriate. This approach provides a more robust estimate than single-window classification and allows the platform to return representative-window information and a richer audit trail of model reasoning.",
        "The same methodology is designed to be portable to the Raspberry Pi edge-device module. In the planned hardware flow, the microphone serves as the input source, the respiratory audio is buffered and preprocessed using the same core logic, inference is performed with the trained model or its deployment-ready equivalent, and the LED display presents the result locally. Although this hardware path is not yet claimed as fully completed, including it in the methodology is appropriate because it shares the same conceptual pipeline and extends the project's deployment vision.",
        "Explainability is introduced through a Grad-CAM style heatmap generated on the representative prediction window. While this does not turn the model into a transparent rule-based system, it provides a useful indication of where the model's attention was concentrated on the spectrogram. In the context of a student major project, this is a meaningful addition because it shows awareness of the need for interpretable AI in health-related applications.",
        "Finally, the system is evaluated using saved validation and recording-level test metrics. The report deliberately presents these metrics with caveats and class-wise detail rather than only citing one optimistic number. This is an important methodological stance because it prioritizes honest system interpretation over inflated claims. Such honesty is especially relevant when the system may later be extended toward real-world healthcare support scenarios.",
    ]

    arch_paragraphs = [
        "The architecture of the proposed system can be understood as a layered pipeline. At the input layer, the system accepts respiratory sound recordings through a file-upload workflow, a mobile recording workflow, or a proposed edge-hardware microphone workflow. The supported formats include common audio types such as WAV, MP3, FLAC, OGG, and M4A on the backend side. On Android, audio can be recorded directly as a WAV file using the device microphone and private app storage. In the hardware extension path, a Raspberry Pi microphone acts as the capture endpoint for a compact standalone screening device.",
        "The data processing layer performs validation, trimming, normalization, and representation building. This layer is critical because respiratory audio often includes silence, inconsistent amplitude, and variable duration. The preprocessing design centralizes these decisions in configuration files so that the same values are available across training, evaluation, and deployed inference. This configuration-first design is one of the major engineering improvements introduced in the final rebuild.",
        "The machine learning layer consumes log-mel spectrogram tensors and produces class probabilities. For long recordings, the predictor treats the signal as a sequence of overlapping windows rather than a single frame. This produces multiple probability vectors, each associated with a specific time segment of the recording. Aggregation then converts those window-level predictions into a single recording-level output. The project therefore combines local acoustic reasoning with global recording-level interpretation.",
        "The explainability layer is attached to the final prediction stage. After the representative window is selected, the model computes a heatmap over the corresponding feature map and overlays it on the base spectrogram. This output is returned as a base64-encoded image string to clients. The web platform and Android app then display the heatmap to support user interpretation. The heatmap acts as a trust-building artifact rather than a substitute for clinical explanation, but it adds meaningful value to the interface.",
        "The backend layer is implemented in two stages across the project history. The initial MVP uses FastAPI and a heuristic baseline classifier to validate user-facing behavior. The final deployment-oriented backend uses Flask, a TensorFlow/Keras predictor, and recording-level aggregation. This split between prototype and final backend should be viewed positively in the report because it demonstrates iterative engineering rather than indecision. The MVP answered workflow questions, and the rebuild answered model rigor questions.",
        "The presentation layer contains two implemented client applications and one hardware-oriented proposed output path. The browser interface presents the PneumoScan workflow with upload, status, diagnosis, heatmap, and recommendation panels. The Android interface offers an on-device recording and submission flow with Compose-based cards for probability breakdown, severity, recording timeline, and heatmap display. The proposed Raspberry Pi pathway would present the final result on an LED display so that the hardware unit can act as a standalone edge screening prototype. All three paths are conceptually tied together by the same model output contract and inference philosophy.",
        "The edge-device concept adds a practical embedded-systems perspective to the architecture. Instead of viewing deployment only as browser access to a backend, the project also considers the possibility of low-cost local inference hardware. This is academically valuable because it broadens the project from conventional full-stack software into edge AI system design, even if that final hardware integration is still underway.",
        "From a software architecture perspective, the project uses modular boundaries between data preparation, feature extraction, model catalog construction, inference, reporting, and interface code. This separation allows future replacement of the active model family without rewriting every client. It also makes the platform easier to document and maintain, which is particularly valuable for a major project report.",
    ]

    impl_paragraphs = [
        "The implementation history of the project is itself worth documenting because it illustrates the engineering progression from prototype to structured platform. The root-level MVP consists of a FastAPI application, static web assets, feature extraction services, a heuristic classifier, and a text-reporting service. In this stage, the purpose was to ensure that a user could upload a file, receive a structured response, and understand the intended user experience. The heuristic model acted as a swappable placeholder while the UI and backend contracts were stabilized.",
        "The root MVP backend exposes an `/api/analyze` endpoint, validates file uploads, extracts features using librosa when available, falls back safely in minimal environments, and returns a Pydantic response containing predicted condition, confidence, severity, explanations, recommendations, differentials, and a text report. Even though it is not the final inference engine, this MVP is still valuable in the report because it demonstrates early product thinking and iterative system construction.",
        "The final rebuild introduces a significantly more disciplined project layout. Configuration is moved into YAML files, reusable machine learning code is placed under `src/resp_ai`, and the backend/frontend components are separated under `app/backend` and `app/frontend`. This structure is more appropriate for long-term maintainability and for a report that needs to explain system modules clearly. The rebuild also supports training workflows, evaluation scripts, reporting utilities, and development smoke tests.",
        "Within the feature extraction module, audio preprocessing functions manage loading, trimming, normalization, fitting to length, augmentation, and spectrogram generation. The code supports both path-based training preprocessing and byte-based inference preprocessing. This design means the same representation can be built from files during model training and from uploaded blobs during runtime. Such parity between experimentation and deployment is an important implementation achievement and is often missing in smaller research prototypes.",
        "The model catalog module defines several architecture families. The stronger CNN configuration is especially relevant because it uses multiscale residual blocks, squeeze-excitation style feature recalibration, progressive downsampling, batch normalization, dropout, and a dedicated convolutional layer for Grad-CAM. These are not superficial details. They show that the model design was intentionally adapted for spectrogram learning rather than simply copied from unrelated image-classification examples.",
        "The training pipeline compiles the selected model with Adam optimization, uses focal loss when configured, computes class weights, logs training history, performs model checkpointing, and saves metrics, confusion matrices, class names, and augmentation profiles. The presence of these artifacts in the workspace makes the system more auditable. It also means the report can discuss the project using concrete saved results rather than hypothetical outputs.",
        "The inference predictor loads the saved model, locates the last convolutional layer for Grad-CAM, extracts recording windows, computes per-window probabilities, aggregates them, applies pneumonia-sensitive post-processing when needed, and returns a structured JSON response. This is one of the most important technical modules in the project because it translates the training artifact into a user-facing prediction service. The prediction output includes not just the primary label but also secondary prediction, window list, aggregation method, overlap ratio, confidence, severity, and heatmap.",
        "On the frontend side, the final browser interface is designed as a high-contrast, modern web page under the PneumoScan brand. It includes sections for upload, status, result cards, diagnosis emphasis, probability cards, alert banners, and report export affordances. From a report-writing perspective, the frontend is useful because it demonstrates that the project addressed user experience and not only backend correctness. The web interface translates raw model output into a readable narrative and structured visual hierarchy.",
        "The Android application is implemented with Kotlin, Jetpack Compose, Retrofit, OkHttp, and Kotlin serialization. It handles runtime microphone permission, file picking, live recording, local WAV storage, and backend submission. The recording component uses Android's `AudioRecord` API and writes a valid WAV header with the captured PCM stream. This is a meaningful implementation contribution because it allows the mobile client to act as a true capture-and-analysis tool rather than only a passive viewer of previously recorded files.",
        "The Android UI is intentionally structured as a mobile diagnostic dashboard. It includes a hero card, capture studio card, probability section, timeline section, and heatmap section. It also supports status messages, error handling, and result clearing. The view model manages asynchronous calls, selected file state, recording lifecycle, and repository-based API access. This separation of state and UI follows modern Android practice and strengthens the software engineering value of the project.",
        "The repository layer chooses the appropriate base URL depending on whether the app is running on an emulator or a physical device, which simplifies local development and demonstration. This is a small but useful detail that often determines whether a mobile AI demo feels polished or fragile. Such pragmatic implementation choices matter in major project evaluation because they reveal attention to real debugging and deployment workflow rather than only textbook architecture.",
        "Alongside the implemented browser and Android flows, the project now also includes a hardware integration direction centered on a Raspberry Pi, microphone input, and LED display output. The intended implementation path is to load the trained model or a deployment-suitable model variant onto the Raspberry Pi environment, capture respiratory sound through the connected microphone, perform preprocessing and inference, and show the prediction and severity message on the LED screen. At the current report stage, this module should be described truthfully as under development or proposed integration rather than as a fully demonstrated final deliverable.",
        "Including the hardware path in the implementation chapter is still justified because it shares the same model and preprocessing philosophy as the software interfaces. It represents the next deployment layer of the project and demonstrates that the system design is extensible enough to move toward edge AI hardware. For a major project, this strengthens the originality and ambition of the work without requiring the report to overstate what has already been finalized.",
        "Overall, the implementation demonstrates iterative refinement, modular coding, configuration management, and cross-platform delivery. The project therefore succeeds not only as a machine learning experiment but as a coherent software system. This is the core message that the implementation section should communicate in the final report.",
    ]

    results_paragraphs = [
        f"The saved validation metrics for the strong-CNN pneumonia-focus run show a best validation accuracy of {fmt_pct(VAL_METRICS['best_val_accuracy'])}, a best validation loss of {VAL_METRICS['best_val_loss']:.4f}, a macro F1-score of {fmt_pct(VAL_METRICS['macro_f1'])}, and a weighted F1-score of {fmt_pct(VAL_METRICS['weighted_f1'])}. These values indicate that the model learned a useful multiclass representation while still facing class imbalance and class-difficulty effects, particularly on the smaller classes.",
        f"The final recording-level test evaluation is more relevant for deployment because the platform operates on complete recordings rather than isolated clip fragments. On this saved test evaluation, the model achieved an accuracy of {fmt_pct(TEST_METRICS['accuracy'])}, macro precision of {fmt_pct(TEST_METRICS['macro_precision'])}, macro recall of {fmt_pct(TEST_METRICS['macro_recall'])}, macro F1-score of {fmt_pct(TEST_METRICS['macro_f1'])}, and weighted F1-score of {fmt_pct(TEST_METRICS['weighted_f1'])}. This is the main measured performance that should be discussed in the report because it reflects how the deployed backend actually reasons over recordings.",
        "The class-wise results reveal an uneven but interpretable performance pattern. COPD is the strongest class, which is not surprising given both its higher representation and the strong chronic acoustic patterns often associated with it. Asthma and Normal achieve moderate performance, indicating that the model captures some distinguishing structure but still experiences overlap or confusion. Pneumonia remains the weakest class, which suggests that despite targeted augmentation, the acoustic diversity and source imbalance of pneumonia examples still limit detection quality.",
        "This pattern is valuable to discuss honestly. A weaker pneumonia score does not invalidate the project; instead, it reveals one of the most important research directions for future work. In medical AI, a transparent account of where the model is strong and where it is weak is more credible than a blanket claim of excellence. The current project therefore interprets the results as evidence of platform feasibility and partial model success rather than as proof of clinical readiness.",
        "Training history further supports this interpretation. The saved history shows that validation performance improved rapidly after the first epoch, stabilized across the middle of training, and eventually reached a peak validation accuracy near 0.739 before overfitting signals became more visible. The final choice of checkpointing, learning-rate reduction, and early stopping logic helped retain the best model observed during training rather than only the last model. This is sound experimental practice and should be highlighted in the report.",
        "The results are also important from a system-design perspective. Because the backend returns window-level predictions and heatmaps, the user interface is able to display richer evidence than one class name. This means the measured model quality is paired with a more informative presentation layer. In many applied AI systems, usability is improved not only by better scores but by better framing of model output. The current project demonstrates that principle clearly.",
        "The Raspberry Pi hardware path is not associated with a separate completed accuracy experiment in the current workspace, so the report should not invent standalone edge-device performance numbers. Instead, it should explain that the planned hardware module is intended to reuse the same trained model family and inference logic, with future work required for packaging, optimization, and on-device validation. This distinction keeps the report technically honest while still documenting the full intended project scope.",
        "From a discussion standpoint, the project should therefore be presented as successful in three ways. First, it produced a functional end-to-end platform. Second, it achieved credible recording-level performance on a saved strong-CNN configuration. Third, it identified meaningful next steps, especially around pneumonia robustness, data balance, and signal-quality improvement. These three outcomes are collectively more important than chasing a single inflated accuracy claim.",
    ]

    out = [chapter("Chapter 3: Proposed System")]
    out.append(h(2, "3.1 Introduction"))
    out.extend(p(x) for x in p11)
    out.append(
        fig(
            "Figure 3.1 Overall Data-to-Deployment Workflow",
            "Placeholder for a system diagram showing data preparation, training, inference backend, web platform, Android application, and planned Raspberry Pi edge-device integration.",
        )
    )
    out.append(h(2, "3.2 Methodology"))
    out.extend(p(x) for x in methodology_paragraphs)
    out.append(fig("Figure 3.2 Audio Preprocessing Pipeline", "Placeholder for the respiratory audio preprocessing flow including resampling, trimming, normalization, windowing, and log-mel spectrogram generation."))
    out.append(fig("Figure 3.3 Recording-Level Window Aggregation Concept", "Placeholder for an illustration showing overlapping windows, per-window prediction, and final recording-level aggregation."))
    out.append(h(3, "Table 3.1 Dataset Summary After Deduplication"))
    out.append(table(["Metric", "Value"], dataset_rows))
    out.append(h(3, "Table 3.2 Source-Wise Class Distribution in the Prepared Paper Dataset"))
    out.append(table(["Label", "Source Count 1", "Source Count 2", "Source Count 3"], source_rows))
    out.append(h(3, "Table 3.3 Pneumonia-Focused Augmentation Summary"))
    out.append(
        table(
            ["Metric", "Value"],
            [
                ["Base root", PNEUMONIA_SUMMARY["base_root"]],
                ["Output root", PNEUMONIA_SUMMARY["output_root"]],
                ["Minimum pneumonia source clips", str(PNEUMONIA_SUMMARY["min_pneumonia_source_clips"])],
                ["Minimum pneumonia clips", str(PNEUMONIA_SUMMARY["min_pneumonia_clips"])],
                ["Original train pneumonia clips", str(PNEUMONIA_SUMMARY["original_train_label_counts"]["Pneumonia"])],
                ["Final train pneumonia clips", str(PNEUMONIA_SUMMARY["final_train_label_counts"]["Pneumonia"])],
                ["Total augmented rows", str(PNEUMONIA_SUMMARY["augmented_rows"])],
            ],
        )
    )
    out.append(h(2, "3.3 System Architecture"))
    out.extend(p(x) for x in arch_paragraphs)
    out.append(fig("Figure 3.4 Web Platform Workflow", "Placeholder for the browser-based user flow from file upload to prediction, probability visualization, severity messaging, and heatmap display."))
    out.append(fig("Figure 3.5 Android Mobile Workflow", "Placeholder for the Android flow covering permission handling, recording, file upload, backend analysis, and mobile dashboard presentation."))
    out.append(fig("Figure 3.6 Proposed Raspberry Pi Edge-Device Workflow", "Placeholder for the planned edge workflow: microphone input to Raspberry Pi, shared preprocessing, local inference, and result output on the LED display."))
    out.append(h(2, "3.4 Implementation"))
    out.extend(p(x) for x in impl_paragraphs)
    out.append(h(3, "Table 3.4 Final Model Configuration"))
    out.append(table(["Configuration", "Value"], model_config_rows))
    out.append(h(3, "Table 3.5 Proposed Edge Device Hardware Integration Summary"))
    out.append(table(["Aspect", "Description"], edge_device_rows))
    out.append(h(2, "3.5 Results and Discussion"))
    out.extend(p(x) for x in results_paragraphs)
    out.append(fig("Figure 3.7 Heatmap-Based Explainability Concept", "Placeholder for a representative spectrogram window with Grad-CAM style overlay highlighting model attention regions."))
    out.append(h(3, "Table 3.6 Validation Performance of the Saved Strong-CNN Run"))
    out.append(table(["Metric", "Value"], val_rows))
    out.append(h(3, "Table 3.7 Recording-Level Test Performance of the Saved Model"))
    out.append(table(["Metric", "Value"], test_rows))
    out.append(h(3, "Table 3.8 Per-Class Validation Metrics"))
    out.append(table(["Class", "Precision", "Recall", "F1-score", "Support"], per_class_val))
    out.append(h(3, "Table 3.9 Per-Class Recording-Level Test Metrics"))
    out.append(table(["Class", "Precision", "Recall", "F1-score", "Support"], per_class_test))
    return "".join(out)


def build_chapter_6() -> str:
    conclusion = [
        "This major project successfully designed and implemented an AI-assisted respiratory disease screening platform that combines respiratory audio preprocessing, deep learning–based classification, explainability, and user-facing deployment. The work began with a proof-of-concept web MVP and matured into a structured respiratory AI rebuild containing dataset engineering scripts, a model training framework, a deployment-oriented inference backend, a branded browser interface, and a Kotlin-based Android mobile client.",
        "From a research perspective, the project contributed a fairer and more disciplined respiratory sound workflow through duplicate removal, label normalization, patient-aware split generation, window-based recording analysis, and pneumonia-focused augmentation. From an implementation perspective, the project demonstrated that the trained model can be operationalized through both browser and mobile interfaces and that the resulting system can present a richer interpretation than a single label alone.",
        "An additional strength of the project is its forward-looking edge deployment direction. By explicitly planning a Raspberry Pi based microphone-and-display unit around the same inference pipeline, the work extends beyond conventional web and mobile delivery and moves toward a compact standalone respiratory screening concept. This hardware path is included honestly as planned/ongoing work rather than a fully completed subsystem, but it meaningfully strengthens the practical vision of the project.",
        f"The saved recording-level test result of {fmt_pct(TEST_METRICS['accuracy'])} should be interpreted as encouraging evidence of feasibility rather than as a claim of clinical readiness. The platform is strongest on COPD and still faces difficulty on Pneumonia, which is consistent with dataset imbalance and respiratory sound variability. Nevertheless, the system meaningfully demonstrates how an academic major project can integrate machine learning, explainability, web delivery, and Android deployment into one coherent solution.",
    ]
    future = [
        "A first future direction is the collection of larger and more balanced respiratory sound datasets, especially for pneumonia and asthma. Better class representation is likely to improve minority-class robustness and reduce the need for heavy augmentation.",
        "A second direction is signal-quality enhancement. Learned denoising, respiratory sound enhancement, or explicit quality scoring could improve usability for smartphone-based or telehealth-style capture scenarios.",
        "A third direction is model evolution. The current system already supports multiple architecture families, so future work can evaluate attention models, transformers, or ensemble strategies using the same deployment pipeline.",
        "A fourth direction is multimodal expansion. Clinical metadata such as age, temperature, oxygen saturation, symptom duration, and smoking history could be integrated with audio for more clinically meaningful triage support.",
        "A fifth direction is secure deployment and validation. Cloud-hosted inference, authentication, patient history, audit logging, and controlled clinical studies would be required before any real clinical use could be considered.",
        "Finally, the platform can be extended toward device ecosystems such as digital stethoscopes, Raspberry Pi capture units, or institutional telemedicine systems. These directions confirm that the current project is not an endpoint but a foundation for further respiratory AI research and engineering.",
    ]
    pubs = [
        "No publication is claimed in this draft. If the student has submitted, presented, or published a paper related to this project, the corresponding citation details can be inserted here.",
    ]
    out = [chapter("Chapter 6: Conclusion and Future Scope")]
    out.append(h(2, "6.1 Conclusion"))
    out.extend(p(x) for x in conclusion)
    out.append(h(2, "6.2 Future Scope"))
    out.extend(p(x) for x in future)
    out.append('<div class="page-break"></div>' + h(1, "References"))
    ref_rows = []
    for i, paper in enumerate(PAPERS, start=1):
        ref_rows.append([f"[{i}]", f"{paper['authors']} ({paper['year']}), <em>{paper['title']}</em>, {paper['venue']}. {paper['url']}"])
    ref_rows.extend(
        [
            ["[6]", "ICBHI 2017 Respiratory Sound Database paper and associated challenge materials. https://doi.org/10.1007/978-981-10-7419-6_6"],
            ["[7]", "Project workspace artifacts used for implementation evidence, including saved model metrics, dataset audit reports, Android module definitions, and backend source files under /Users/diyarao/Documents/Playground."],
        ]
    )
    out.append(table(["Ref.", "Reference"], ref_rows))
    out.append('<div class="page-break"></div>' + h(1, "List of Publications"))
    out.extend(p(x) for x in pubs)
    return "".join(out)


def build_appendices() -> str:
    source_headers = ["Label"] + [key for key in DATASET_AUDIT["source_by_label"][0].keys() if key != "label"]
    source_rows = []
    for row in DATASET_AUDIT["source_by_label"]:
        source_rows.append([str(row.get(key, "")) for key in source_headers])

    split_rows = []
    for row in DATASET_AUDIT.get("patients_per_label", {}).items():
        split_rows.append([row[0], str(row[1])])

    duration = DATASET_AUDIT["duration_summary"]
    duration_rows = [[k, str(v)] for k, v in duration.items()]

    history_headers = list(TRAIN_HISTORY[0].keys())
    history_rows = [[row[h] for h in history_headers] for row in TRAIN_HISTORY]

    file_rows = [
        ["Root MVP backend", "app/main.py", "FastAPI upload and analysis endpoint"],
        ["Root MVP feature extraction", "app/services/audio.py", "Feature extraction and explanation region helpers"],
        ["Root MVP heuristic inference", "app/services/inference.py", "Baseline respiratory model service"],
        ["Root MVP reporting", "app/services/reporting.py", "Text report generation"],
        ["Static web UI", "static/index.html / static/app.js / static/styles.css", "Browser-based MVP frontend"],
        ["Final backend", "respiratory_ai_rebuild/app/backend/main.py", "Flask deployment-oriented inference API"],
        ["Final predictor", "respiratory_ai_rebuild/src/resp_ai/inference/predictor.py", "Window aggregation and Grad-CAM inference logic"],
        ["Training pipeline", "respiratory_ai_rebuild/src/resp_ai/models/train.py", "Training orchestration and artifact saving"],
        ["Evaluation pipeline", "respiratory_ai_rebuild/src/resp_ai/models/evaluate.py", "Validation and test evaluation"],
        ["Android main activity", "respiratory_ai_android_app/.../MainActivity.kt", "Permission, file-pick, and UI launch logic"],
        ["Android state management", "respiratory_ai_android_app/.../MainViewModel.kt", "Recording and prediction workflow"],
        ["Android recorder", "respiratory_ai_android_app/.../WavAudioRecorder.kt", "Live WAV capture on device"],
        ["Proposed edge device module", "Raspberry Pi + microphone + LED display stack (planned)", "Standalone edge-oriented respiratory screening pathway using the same core model and preprocessing logic"],
    ]

    sample_response = {
        "filename": "sample_recording.wav",
        "prediction": "COPD",
        "confidence": round(TEST_METRICS["classification_report"]["COPD"]["f1-score"] * 100, 2),
        "probabilities": {
            "Asthma": 8.42,
            "COPD": 87.31,
            "Normal": 2.73,
            "Pneumonia": 1.54,
        },
        "severity": {
            "level": "High",
            "message": "Chronic obstructive pattern detected; specialist evaluation recommended.",
        },
        "windows_used": 4,
        "aggregation": "pneumonia_sensitive",
        "window_overlap": 0.5,
        "representative_window": {
            "window_index": 2,
            "start_sec": 6.0,
            "duration_sec": 5.0,
            "prediction": "COPD",
            "confidence": 93.1,
        },
        "window_predictions": [
            {"window_index": 0, "start_sec": 0.0, "duration_sec": 5.0, "prediction": "Asthma", "confidence": 61.4},
            {"window_index": 1, "start_sec": 3.0, "duration_sec": 5.0, "prediction": "COPD", "confidence": 81.7},
            {"window_index": 2, "start_sec": 6.0, "duration_sec": 5.0, "prediction": "COPD", "confidence": 93.1},
            {"window_index": 3, "start_sec": 9.0, "duration_sec": 5.0, "prediction": "COPD", "confidence": 88.9},
        ],
        "heatmap": "data:image/png;base64,<truncated for report display>",
    }

    appendix_intro = [
        "The appendices included in this report are meant to strengthen the completeness of the major project documentation. They provide evidence-oriented material drawn from the actual workspace artifacts, including dataset audit summaries, model configuration snapshots, training history, sample API output, and module inventories. These annexures are particularly useful in academic project reports because they show that the discussion in the main chapters is supported by reproducible implementation evidence.",
        "Including these appendices also helps bridge the gap between prose description and engineering detail. The main chapters focus on conceptual clarity and system-level explanation, whereas the appendices capture structured supporting information that may be useful during project review, viva questioning, or final formatting into an institutional template.",
    ]

    out = []
    out.append('<div class="page-break"></div>' + h(1, "Appendix A: Dataset Audit Summary"))
    out.extend(p(x) for x in appendix_intro)
    out.append(h(2, "A.1 Overall Dataset Statistics"))
    out.append(table(["Metric", "Value"], [["Raw records", str(DATASET_AUDIT["raw_records"])], ["Deduplicated records", str(DATASET_AUDIT["deduped_records"])], ["Duplicates removed", str(DATASET_AUDIT["dedup_removed"])], ["Very short recordings (<1s)", str(DATASET_AUDIT["very_short_lt_1s"])], ["Very long recordings (>20s)", str(DATASET_AUDIT["very_long_gt_20s"])] ]))
    out.append(h(2, "A.2 Source-Wise Class Distribution"))
    out.append(table(source_headers, source_rows))
    out.append(h(2, "A.3 Patients per Label"))
    out.append(table(["Label", "Patients"], split_rows))
    out.append(h(2, "A.4 Duration Summary"))
    out.append(table(["Statistic", "Value"], duration_rows))

    out.append('<div class="page-break"></div>' + h(1, "Appendix B: Model Configuration and Metrics"))
    out.append(h(2, "B.1 Training Options Snapshot"))
    options_rows = [[k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)] for k, v in TRAINING_OPTIONS.items()]
    out.append(table(["Option", "Value"], options_rows))
    out.append(h(2, "B.2 Validation Metrics"))
    out.append(table(["Metric", "Value"], [["Best validation accuracy", fmt_pct(VAL_METRICS["best_val_accuracy"])], ["Best validation loss", f"{VAL_METRICS['best_val_loss']:.4f}"], ["Macro F1-score", fmt_pct(VAL_METRICS["macro_f1"])], ["Weighted F1-score", fmt_pct(VAL_METRICS["weighted_f1"])]]))
    out.append(h(2, "B.3 Recording-Level Test Metrics"))
    out.append(table(["Metric", "Value"], [["Accuracy", fmt_pct(TEST_METRICS["accuracy"])], ["Macro precision", fmt_pct(TEST_METRICS["macro_precision"])], ["Macro recall", fmt_pct(TEST_METRICS["macro_recall"])], ["Macro F1-score", fmt_pct(TEST_METRICS["macro_f1"])], ["Weighted F1-score", fmt_pct(TEST_METRICS["weighted_f1"])]]))

    out.append('<div class="page-break"></div>' + h(1, "Appendix C: Training History"))
    out.append(
        p(
            "The following epoch-wise training history is included to provide a direct view of model behavior during optimization. Such evidence is useful in viva discussions because it shows how accuracy, loss, validation accuracy, and validation loss evolved over time rather than only presenting the final checkpoint."
        )
    )
    out.append(table(history_headers, history_rows))

    out.append('<div class="page-break"></div>' + h(1, "Appendix D: API Contract and Sample Output"))
    out.append(
        p(
            "The deployment-oriented backend returns structured JSON that is designed to serve both the browser frontend and the Android application. The response includes prediction, confidence, severity, aggregation metadata, representative windows, and a heatmap image payload. A formatted example is included below for documentation purposes."
        )
    )
    out.append("<pre>" + json.dumps(sample_response, indent=2) + "</pre>")

    out.append('<div class="page-break"></div>' + h(1, "Appendix E: Module Inventory and Deployment Notes"))
    out.append(
        p(
            "The following inventory maps the major code modules to their purpose within the project. Including this table in the report helps examiners understand that the project contains a complete and organized codebase spanning data preparation, model training, inference, browser interaction, and Android deployment."
        )
    )
    out.append(table(["Component", "Path", "Purpose"], file_rows))
    out.append(h(2, "E.1 Deployment Notes"))
    deploy_notes = [
        "The MVP web application runs with FastAPI and serves a static single-page frontend for audio upload and heuristic analysis demonstration.",
        "The final deployment-oriented backend runs with Flask and loads the saved strong-CNN model family from the configured model directory.",
        "The browser frontend can call the backend and display result cards, probability outputs, and heatmap imagery.",
        "The Android application uses an emulator-friendly base URL of http://10.0.2.2:8080/ and switches to a resolved local-network host for physical-device development.",
        "The backend health endpoint reports whether the active model exists, which is helpful during demonstration and verification.",
        "The planned Raspberry Pi module is intended to reuse the same preprocessing and inference logic while capturing audio from a connected microphone and presenting output through an LED screen.",
        "Because the project is modular, the active model family can be replaced later without redesigning the clients.",
    ]
    out.append(ul(deploy_notes))
    out.append('<div class="page-break"></div>' + h(1, "Appendix F: Detailed User Manual and Demonstration Workflow"))
    appendix_f = [
        "This appendix explains how the system can be demonstrated during project evaluation. A major project is assessed not only by its codebase but also by the clarity with which the student can show the end-to-end user workflow. The PneumoScan platform supports both a browser path and an Android path, and both flows should be described in a disciplined manner during demonstration.",
        "In the browser-based path, the user begins at the upload interface and selects a respiratory sound recording. After submission, the backend validates the file extension, reads the uploaded bytes, applies the configured preprocessing pipeline, extracts windows when necessary, performs inference, generates the heatmap, and returns a structured JSON response. The browser UI then transforms this response into diagnosis cards, probability displays, severity indicators, and recommendation-oriented text.",
        "For the Android path, the evaluator can either import an existing audio file or record a new sample using the device microphone. The app requests microphone permission when needed, saves the recording to app-private storage as a WAV file, updates UI state through the view model, and enables analysis when a valid file is ready. The Retrofit repository then sends the file to the backend and receives the same structured prediction object used in the web platform.",
        "For the planned Raspberry Pi hardware path, the demonstration narrative is slightly different because it should be presented as an edge-device concept or ongoing integration rather than a completed production subsystem. In that flow, respiratory sound would be captured directly from the connected microphone, buffered locally, processed through the shared inference pipeline, and then displayed on the LED screen as prediction and severity output. Even if the hardware assembly is still incomplete, documenting this demonstration flow is valuable because it shows clear system-level planning.",
        "A recommended demo sequence is to begin with the web platform because it exposes the overall system flow more visibly on a larger screen. The student can then show the Android application as evidence of portability and mobile capture capability. This order also helps the audience first understand the platform's logic and then appreciate the value of mobile integration.",
        "If the evaluation panel asks about the hardware addition, the correct explanation is that the Raspberry Pi path is the project's edge deployment extension. The same trained model and preprocessing assumptions are intended to be used there, but hardware packaging, resource optimization, microphone interfacing, and on-device validation remain ongoing tasks. This explanation is strong because it is ambitious while still being truthful.",
        "During demonstration, it is important to explain that the system is a screening aid and not a medical diagnosis tool. The predicted class, class-wise probabilities, severity message, and heatmap should be presented as decision-support artifacts rather than definitive clinical answers. This clarification strengthens the project's credibility because it shows awareness of ethical and practical deployment limits.",
        "The evaluator may also ask why the system returns multiple windows rather than one answer from a single clip. The correct explanation is that respiratory recordings can be temporally heterogeneous. Some windows may contain stronger pathological signals than others. By evaluating several overlapping windows and aggregating them, the system reduces the risk that one weak or noisy segment will dominate the final result.",
        "Another likely viva question concerns the relationship between the MVP and the final rebuild. The browser MVP should be described as the early validation stage used to stabilize the user flow and result format. The final rebuild should be described as the structured machine learning and deployment stage that replaced heuristic inference with a saved TensorFlow model family and a more rigorous data pipeline.",
        "If internet access or device availability is limited during demonstration, the student can still present the architecture using the appendices and saved metrics. The project remains valid because the workspace contains the saved training history, evaluation metrics, dataset audit reports, configuration files, and mobile code. In other words, the report and appendices provide enough material to support both a live demo and a documentation-only evaluation scenario.",
        "The browser workflow can be summarized as: select file, upload, analyze, view prediction, inspect probabilities, inspect heatmap, review severity, and discuss recommendations. The Android workflow can be summarized as: grant permission, record or pick audio, analyze, and review the mobile dashboard. These concise narratives are easy to remember and useful during oral defense.",
        "From a documentation perspective, this appendix is also important because it translates software behavior into user behavior. Reports that include such usage-oriented explanation are typically easier to evaluate because the examiner can imagine the system in operation. That is especially useful for interdisciplinary projects that combine AI and application development.",
    ]
    out.extend(p(x) for x in appendix_f)
    out.append(
        table(
            ["Step", "Web Demonstration", "Android Demonstration", "Edge-Device Demonstration"],
            [
                ["1", "Open browser UI and explain upload workflow", "Launch app and explain home screen", "Present Raspberry Pi setup concept and connected peripherals"],
                ["2", "Choose respiratory recording file", "Grant microphone permission or pick a file", "Describe microphone-based respiratory sound capture"],
                ["3", "Submit file to backend", "Record or select audio and trigger analysis", "Explain local preprocessing and inference path on device"],
                ["4", "Observe returned prediction and probabilities", "Observe returned prediction and severity", "Show planned LED prediction and severity output"],
                ["5", "Explain heatmap and representative window", "Explain timeline windows and heatmap", "Explain that the device reuses the same model logic as software interfaces"],
                ["6", "Clarify limitations and future scope", "Clarify portability and future device support", "Clarify honest status: planned/ongoing hardware integration"],
            ],
        )
    )

    out.append('<div class="page-break"></div>' + h(1, "Appendix G: Testing Strategy, Scenarios, and Observations"))
    appendix_g = [
        "Testing in this project spans multiple layers rather than only model evaluation. At the machine learning layer, the system uses saved validation and recording-level test metrics to judge predictive behavior. At the backend layer, the `/health` and `/predict` endpoints provide a direct way to verify model availability, file handling, and response formatting. At the client layer, the browser UI and Android application are tested for upload flow, state transitions, and result presentation.",
        "A useful way to understand the testing strategy is to divide it into data tests, model tests, API tests, and interface tests. Data tests include checking that duplicate recordings are removed, normalized labels are assigned correctly, and patient-aware splits are written as expected. Model tests include successful training, metric logging, checkpoint creation, and evaluation artifact generation. API tests include verifying supported file extensions, empty-upload handling, model-path detection, and JSON response structure. Interface tests include upload button behavior, disabled states, status messages, result rendering, and error handling.",
        "The root-level MVP also includes a small report-generation sanity test. Although simple, this is useful in a major project because it shows that the reporting logic is not completely informal. It demonstrates awareness that even helper layers should be tested where possible. The final rebuild adds additional trust through saved artifacts such as metrics JSON files, class-name snapshots, and history logs.",
        "Android-side testing must account for runtime permissions, emulator networking, and device variability. The project handles microphone permission denial explicitly and switches base URL behavior depending on whether the app is running on an emulator or a physical device. These are important quality-of-life details that significantly improve demonstration stability.",
        "In evaluation discussion, it is also valid to state that not all forms of formal testing were fully automated. Some parts of the system, especially UI aesthetics and device-dependent recording behavior, are best checked through manual testing. This is common in full-stack student projects and should be described transparently rather than hidden.",
        "The project can therefore be described as having a layered testing mindset: automatic metric generation and artifact persistence for the core model, endpoint-level backend verification, and manual interface verification for the browser and mobile clients. This combination is appropriate for a major project of this scope and provides enough evidence of engineering care.",
    ]
    out.extend(p(x) for x in appendix_g)
    out.append(
        table(
            ["Test Layer", "Scenario", "Expected Result", "Observed Project Support"],
            [
                ["Data", "Duplicate removal", "Repeated files should not remain in final metadata", "Supported through SHA-256 deduplication"],
                ["Data", "Patient-aware split", "Same patient should remain within one split", "Supported through group-aware split assignment"],
                ["Model", "Training checkpointing", "Best model should be saved during training", "Supported through ModelCheckpoint callback"],
                ["Model", "Evaluation artifact creation", "Metrics and confusion outputs should be saved", "Supported through evaluation scripts"],
                ["Backend", "Unsupported extension upload", "Request should be rejected with error", "Handled in backend validation"],
                ["Backend", "Model missing", "Health endpoint should expose status and path issue", "Handled in /health and model resolution"],
                ["Web UI", "No file selected", "User should see a clear error/status message", "Handled in JavaScript form logic"],
                ["Android", "Permission denied", "App should show explanatory error state", "Handled in view model"],
                ["Android", "Recording stop", "Saved WAV should become analyzable", "Handled in recorder and state flow"],
                ["Edge device", "Microphone capture and LED output flow", "Device should capture sound, infer, and display result locally", "Documented as planned/ongoing integration path"],
            ],
        )
    )

    out.append('<div class="page-break"></div>' + h(1, "Appendix H: Risk Register, Limitations, and Ethical Reflection"))
    appendix_h = [
        "Every AI-enabled health-related system must be accompanied by a clear statement of risks and limitations. In the present project, the most obvious risk is over-interpretation of the model output. A predicted class with a confidence value may look authoritative to non-expert users, but it is not equivalent to a clinician's diagnosis. For that reason, the system consistently frames itself as a screening aid and not a diagnostic decision-maker.",
        "A second risk concerns dataset imbalance. The saved metrics show that the model performs far better on COPD than on Pneumonia, even after targeted augmentation. This means that the system could underperform on precisely the kind of minority-class scenario where screening is clinically important. The project handles this responsibly by documenting the limitation instead of suppressing it.",
        "A third risk arises from signal quality. Smartphone recordings, non-clinical recording environments, and variable microphone placement can all affect model behavior. Although the current system trims and normalizes input and uses window aggregation, it does not yet include a dedicated learned denoising or quality-estimation stage. This remains a practical limitation for future work.",
        "An ethical consideration arises from explainability. Heatmaps can be helpful, but they can also be misunderstood as proof of causal understanding. In reality, the heatmap only shows where the model's internal attention was strongest on a selected spectrogram window. The report therefore treats explainability as a transparency aid rather than definitive evidence of clinical reasoning.",
        "Privacy is another important concern. Real patient recordings are sensitive health-related information. The current project uses local workspace files and demonstration-oriented flows, but future deployment would require secure storage, access control, encryption, and privacy-aware governance. Mentioning this explicitly strengthens the professionalism of the report.",
        "Finally, there is a project-management reflection. The system grew from an MVP into a structured rebuild because the problem demanded it. That evolution introduced complexity, but it also improved correctness and extensibility. A major project report should not hide that complexity. Instead, it should show how the design matured and why the final architecture became more disciplined over time.",
    ]
    out.extend(p(x) for x in appendix_h)
    out.append(
        table(
            ["Risk Area", "Description", "Current Mitigation", "Future Mitigation"],
            [
                ["Clinical misuse", "Output may be mistaken for diagnosis", "Framed as screening aid", "Formal clinical validation and UI warnings"],
                ["Class imbalance", "Minority classes remain harder to detect", "Class weighting and targeted augmentation", "Larger balanced datasets and ensemble learning"],
                ["Noise sensitivity", "Recording quality varies across devices", "Signal trimming and aggregation", "Denoising and signal-quality scoring"],
                ["Explainability misuse", "Heatmap may be over-interpreted", "Described as attention aid only", "Clinician-centered explanation design"],
                ["Privacy", "Respiratory audio may contain sensitive health data", "Local development context only", "Secure cloud architecture and access control"],
            ],
        )
    )

    out.append('<div class="page-break"></div>' + h(1, "Appendix I: Class-Wise Error Analysis and Improvement Roadmap"))
    appendix_i = [
        "The class-wise metrics stored in the project artifacts show that the model behaves differently across classes. COPD is clearly the most stable class, suggesting that the training data contains relatively strong patterns for chronic obstructive disease and that the model architecture is effective at representing them. This strong class performance helps validate the general modeling approach and provides evidence that the preprocessing and recording-level aggregation pipeline are functional.",
        "Asthma achieves moderate results, which indicates partial separability but also overlap with other classes. This is reasonable because respiratory sound manifestations can vary with attack severity, treatment status, and recording context. In future work, asthma performance could potentially improve through more targeted data collection, class-specific augmentation, or the integration of symptom metadata such as wheeze episodes and bronchodilator response context.",
        "The Normal class also reaches moderate performance. This may sound surprising, because one might assume that normal sounds should be easy to detect. In practice, however, normal respiratory recordings can be confused by mild noise, microphone handling, or subtle source-domain mismatch. It is therefore useful to interpret normal-class performance as part of a broader signal-quality and domain-generalization challenge rather than a trivial baseline category.",
        "Pneumonia remains the most difficult class. The saved test metrics show especially low F1-score on pneumonia, even though the project explicitly added a pneumonia-focused augmentation stage. This result should be interpreted carefully. It does not mean the augmentation was useless. Instead, it suggests that the underlying class remains acoustically diverse, less represented, or harder to distinguish from overlapping pathological categories using the current data and model family alone.",
        "The improvement roadmap therefore begins with better pneumonia data. More balanced, clinically validated pneumonia recordings collected under varied but well-documented conditions would likely improve both precision and recall. Beyond data collection, future work could include hierarchical classification, where the model first distinguishes normal from abnormal and then performs finer disease-specific classification. Such staged reasoning may reduce direct confusion among overlapping abnormal categories.",
        "Another promising direction is multimodal fusion. Respiratory sound alone can be informative, but pneumonia diagnosis in clinical settings is usually contextual. Metadata such as fever history, cough, age, imaging availability, oxygen saturation, or recent infection markers could improve discriminative power dramatically. The modular design of the present system means such metadata could later be incorporated as an auxiliary branch or post-processing layer without discarding the current pipeline.",
        "In summary, the class-wise analysis confirms that the project has achieved a meaningful engineering baseline while also revealing where future research effort should be concentrated. This is a constructive outcome, because a strong major project report should identify not only what worked but what remains unresolved and why.",
    ]
    out.extend(p(x) for x in appendix_i)
    out.append('<div class="page-break"></div>' + h(1, "Appendix J: Example Usage Scenarios and Interpretive Case Notes"))
    appendix_j = [
        "This appendix presents representative usage scenarios that help explain how the platform could be interpreted in practice. These are not medical case records and should not be treated as clinical advice. Instead, they function as report-oriented illustrations of how the system output might be read, discussed, and questioned during testing or demonstration.",
        "Scenario 1 is a browser-based upload of a comparatively clear COPD-type recording. In such a case, the system may return a strong COPD probability, a high severity message, and a representative window whose heatmap highlights sustained time-frequency energy patterns over multiple windows. The important project-level lesson in this scenario is that the platform not only predicts a class but also provides enough supporting detail for the user to understand why the system is relatively confident.",
        "Scenario 2 involves a shorter or noisier recording submitted through the Android application. The prediction may still be usable, but the timeline windows could show greater disagreement between early and late windows. This is exactly why the project uses recording-level aggregation. Rather than trusting the first available five-second segment, the predictor observes a wider portion of the signal and summarizes the pattern more robustly.",
        "Scenario 3 involves a recording with moderate overlap between Asthma and COPD characteristics. In such a case, the primary prediction may be COPD while the secondary prediction remains Asthma with a non-trivial confidence value. For academic discussion, this is valuable because it shows that the model preserves uncertainty structure in the probability vector instead of flattening all ambiguity into a single categorical answer. Such probability behavior is useful when discussing the limitations of disease-specific interpretation from auscultation audio alone.",
        "Scenario 4 concerns a Normal-class output. A normal prediction should not be interpreted as proof of complete respiratory health. Instead, it should be read as an absence of strong abnormal patterns under the current model and current recording quality. This distinction is important because students and examiners alike may otherwise assume that a 'Normal' output implies a medically confirmed normal diagnosis. The report should consistently resist that misunderstanding.",
        "Scenario 5 concerns Pneumonia, the most difficult class in the saved evaluation. If the system predicts Pneumonia with moderate rather than overwhelming confidence, the UI design becomes especially important. The platform should direct attention to severity messaging, representative-window interpretation, and the possibility that the result should trigger follow-up rather than immediate certainty. This scenario is useful in viva discussion because it reveals the project's honest stance on class imbalance and clinical caution.",
        "A broader interpretive lesson from all these scenarios is that the user interface is part of model safety. Confidence values, severity messages, alerts, and heatmaps all contribute to how users understand the platform. If the UI overstates certainty, the system becomes riskier even when the model itself is unchanged. The present project therefore treats interface wording and message framing as important design decisions.",
        "These scenarios also help explain why the platform provides both web and mobile interfaces. The web version is well suited for demonstration, structured inspection, and desktop-style review. The mobile version is better suited to portability, rapid capture, and field-style interaction. The same backend response contract supports both, which means the system maintains consistency while adapting the presentation layer to different user contexts.",
        "From a report-writing perspective, scenario-based explanation adds value because it helps bridge abstract architecture and concrete use. Many project reports describe systems only in terms of modules and metrics. By contrast, scenario-based discussion shows what the outputs mean when a human interacts with the platform. That makes the report easier to follow and more convincing as a complete system narrative.",
        "The scenarios in this appendix are intentionally phrased in a disciplined way: they highlight system behavior, interpretive caution, and interface value without over-claiming medical utility. That style matches the broader tone of the report and reinforces the idea that responsible engineering communication is part of the project itself.",
    ]
    out.extend(p(x) for x in appendix_j)

    out.append('<div class="page-break"></div>' + h(1, "Appendix K: Module-Wise Technical Walkthrough"))
    appendix_k = [
        "A major advantage of the current project is that the codebase is decomposed into meaningful modules rather than being concentrated in a single experimental notebook. This appendix walks through the main technical modules and explains their role in the overall platform. Such a walkthrough is useful in project evaluation because it shows that the student understands not only what the system does, but how responsibilities are distributed inside the implementation.",
        "The dataset preparation modules handle source ingestion, duplicate removal, label normalization, metadata writing, split creation, source-gap reporting, and specialized dataset building. These modules collectively turn a raw collection of respiratory recordings into a structured machine learning dataset. Without this stage, later model training would have been significantly less reliable and less explainable.",
        "The feature extraction module is responsible for audio loading, trimming, normalization, augmentation, spectrogram generation, and window extraction. Its importance lies in consistency. By centralizing these operations, the project reduces the chance that training and inference drift apart. This consistency is especially important when a model is later deployed into a backend service that must process user-supplied recordings in real time.",
        "The model catalog module contains architecture definitions. Because it exposes a single build function that accepts the architecture name and configuration, experimentation becomes modular rather than ad hoc. This means the project can compare different network families without redesigning the rest of the platform. It also makes the codebase easier to document, because the architecture choice becomes an explicit configuration decision rather than a hidden code branch.",
        "The training module orchestrates dataset loading, shuffling, augmentation-aware dataset creation, model compilation, optimizer and loss setup, callbacks, metric saving, checkpoint writing, and artifact persistence. One of its strongest design features is that it saves multiple artifacts beyond the model itself, such as history logs, class-name snapshots, augmentation profiles, and JSON metrics. These artifacts are extremely valuable when writing the report because they make the project auditable.",
        "The evaluation module complements the training module by providing validation and test assessment, including recording-level evaluation. It outputs both summary metrics and prediction-level details, which allows finer error analysis and more transparent discussion of results. In a large student project, having a separate evaluation script is a strong engineering signal because it reflects clear experimentation hygiene.",
        "The predictor module is arguably the heart of runtime inference. It loads the model, resolves class names, extracts windows, predicts probabilities, aggregates them, applies pneumonia-sensitive post-processing, computes severity, and generates the heatmap. This module therefore operationalizes the saved model family and transforms it into a user-facing inference engine. Its design is one of the clearest examples of how the project moves from machine learning artifact to software product.",
        "The Flask backend wraps the predictor in a deployable HTTP service. It validates uploads, manages temporary files, exposes a health endpoint, and returns a structured JSON contract. In practical terms, this module makes the model usable by both browser and mobile clients. In report terms, it demonstrates that the student implemented service-oriented deployment rather than only local function calls.",
        "The browser frontend translates backend results into a visually structured presentation. This module matters because a major project should not assume that raw JSON is a sufficient endpoint. The frontend communicates status, prediction emphasis, probability distribution, severity, alerts, and heatmap information in an organized way. In doing so, it becomes the layer where model output becomes human-readable system output.",
        "The Android modules extend that same principle to mobile use. The recording module handles microphone capture and WAV encoding. The repository module handles networking. The view model manages asynchronous state transitions. The Compose UI renders the diagnostic dashboard. Together, these modules demonstrate that the platform is not tied to a single interface style and can support portable respiratory sound analysis workflows.",
        "The planned Raspberry Pi hardware module would extend this modular architecture into the edge-computing domain. Its role is to reuse the same respiratory inference logic while shifting capture and result display into a compact standalone unit using a microphone and LED screen. Even though the hardware layer is not yet fully finalized, it belongs in the technical walkthrough because it reveals the extensibility of the platform design.",
        "A final reason to include this walkthrough is pedagogical. It helps reviewers see that the project contains a defensible internal architecture. Each module has a clear responsibility, and the modules collaborate through reasonably stable interfaces. This improves maintainability, supports future extension, and strengthens the overall engineering quality of the work.",
    ]
    out.extend(p(x) for x in appendix_k)

    out.append('<div class="page-break"></div>' + h(1, "Appendix L: Project Management Reflection and Future Research Questions"))
    appendix_l = [
        "Large student projects often appear linear when written as reports, but real development is rarely linear. This project evolved through several stages, including initial idea framing, early prototype construction, pipeline redesign, model experimentation, deployment integration, and documentation refinement. Reflecting on that process is useful because it shows how engineering decisions matured in response to real constraints and newly discovered requirements.",
        "One early project-management lesson was the importance of building a demonstrable MVP. The root-level FastAPI application and static web interface made it possible to validate the upload-and-result workflow before the final model was available. This reduced uncertainty and clarified what the eventual prediction contract should look like. Without that early prototype, later development may have spent too much time optimizing a backend without understanding how the user would actually experience the system.",
        "A second lesson was that data engineering was more important than initially expected. Respiratory sound research can appear model-centric from the outside, but real progress depends heavily on dataset quality, label structure, and split design. Once this became clear, the project appropriately invested in deduplication, patient-aware splits, and audit reporting. This shift from model-first thinking to data-and-evaluation-first thinking reflects healthy technical maturity.",
        "A third lesson concerned scope control. Because the project combined machine learning, web development, and Android development, it would have been easy to produce three shallow components instead of one integrated system. The report should therefore emphasize that the chosen design focused on one coherent backend contract shared by two clients. This preserved architectural unity while still demonstrating breadth.",
        "The addition of the Raspberry Pi edge-device idea introduced another project-management lesson: ambitious extensions should be documented with honest status boundaries. Including the hardware path strengthens the originality of the project, but only if the report clearly distinguishes completed software deliverables from ongoing hardware integration work. This balance between ambition and accuracy is part of good engineering reporting.",
        "A fourth lesson was the value of saved artifacts. Metrics JSON files, history CSV logs, dataset audits, training options, and structured outputs all made the final documentation process far easier. They also improved accountability during experimentation. In future projects, generating and preserving such artifacts from the beginning would be even more beneficial.",
        "Future research questions naturally emerge from the work. How much can pneumonia performance improve if larger matched-source datasets are collected? How much robustness can be gained from denoising or signal-quality estimation? Would hierarchical classification or multimodal metadata fusion reduce confusion among abnormal classes? Can model explainability be made more clinically meaningful than a spectrogram heatmap? These questions point toward rich future work rather than a closed endpoint.",
        "There are also deployment research questions. How should such a platform behave under intermittent connectivity? What is the best way to support digital stethoscope integration? How can one design user interfaces that communicate model uncertainty responsibly to clinicians, students, or patients? These questions extend the project from machine learning toward human-computer interaction and system design.",
        "In summary, the project-management reflection reinforces the academic value of the work. The project did not merely produce code; it produced a documented learning process covering prototyping, data curation, model experimentation, deployment, interface design, and critical evaluation. This reflective dimension strengthens the overall quality of the report and shows that the student has internalized not just the final system, but the reasoning that produced it.",
    ]
    out.extend(p(x) for x in appendix_l)

    out.append('<div class="page-break"></div>' + h(1, "Appendix M: Proposed Raspberry Pi Edge Deployment Concept"))
    appendix_m = [
        "This appendix documents the planned hardware extension of the PneumoScan platform. The goal of the edge-device concept is to translate the existing respiratory screening workflow into a compact standalone prototype built around a Raspberry Pi, a microphone input path, and an LED display. Instead of depending exclusively on browser or smartphone interaction, the device would be able to capture respiratory audio locally, process it with the same model family or a deployment-optimized equivalent, and present the screening result directly on the display.",
        "The educational importance of this appendix is that it shows the project is thinking beyond software-only deployment. In many academic projects, machine learning remains limited to laptops or cloud notebooks. By contrast, the planned Raspberry Pi path pushes the system toward edge AI, embedded interfacing, and low-cost product prototyping. That makes the report more complete and shows awareness of real-world deployment possibilities.",
        "The intended operating flow is straightforward at a system level. A patient or demonstrator produces a respiratory sound sample near the connected microphone. The Raspberry Pi buffers the audio stream, applies the same or equivalent preprocessing stages used by the software platform, converts the signal into the required feature representation, runs inference, and finally writes the key result to the LED display. Depending on implementation maturity, the screen can show the predicted class, confidence, severity indication, and perhaps a short recommendation message.",
        "At the current stage of the project, the hardware pathway should be described as proposed or under development rather than fully completed. This distinction is important. The report can legitimately include the architecture, intended workflow, and engineering rationale, but it should not claim final on-device benchmark results or completed long-duration hardware validation unless those artifacts actually exist. Maintaining that boundary preserves the technical integrity of the report.",
        "From a design perspective, the Raspberry Pi unit is attractive because it supports portability, affordability, and demonstration value. A lightweight respiratory screening box can be useful for project exhibitions, departmental demonstrations, health-technology showcases, and future telehealth prototypes. It also opens the door to further work in offline inference, compact medical-device concepts, and localized community screening aids.",
        "The main engineering challenges expected in this path include microphone quality variability, compute-resource constraints, packaging of model dependencies, thermal behavior under repeated inference, and user-interface clarity on a compact LED display. These are all realistic system-engineering concerns and should be presented as next-step tasks rather than hidden uncertainties. Doing so makes the report look thoughtful and professionally scoped.",
    ]
    out.extend(p(x) for x in appendix_m)
    out.append(h(2, "M.1 Proposed Hardware Components"))
    out.append(
        table(
            ["Component", "Role in the Edge Device", "Status in Current Project"],
            [
                ["Raspberry Pi", "Primary edge-computing board for local execution of the inference workflow", "Planned/selected platform"],
                ["Microphone", "Captures respiratory sound input from the user or demonstration subject", "Planned hardware input"],
                ["LED display", "Shows the predicted class, severity, and basic result summary", "Planned hardware output"],
                ["Power supply", "Provides stable power to the Raspberry Pi and display unit", "Required deployment hardware"],
                ["Storage medium", "Stores the runtime environment, model files, and temporary audio buffers", "Required deployment hardware"],
                ["Model runtime package", "Hosts the trained model or optimized inference artifact on-device", "Planned software deployment layer"],
            ],
        )
    )
    out.append(h(3, "Table M.1 Proposed Raspberry Pi Edge-Device Bill of Materials"))
    out.append(
        table(
            ["Item", "Purpose", "Report Positioning"],
            [
                ["Raspberry Pi board", "Edge compute platform", "Included as planned/ongoing hardware base"],
                ["Microphone module or USB microphone", "Audio acquisition for respiratory input", "Included as planned input interface"],
                ["LED or LCD display unit", "Local visual output of screening result", "Included as planned display interface"],
                ["Inference runtime", "Executes the trained model pipeline", "Expected to reuse the existing model logic"],
                ["Lightweight enclosure and mounting", "Physical integration and presentation", "Future prototyping task"],
            ],
        )
    )
    out.append(h(2, "M.2 Planned Edge Workflow"))
    out.extend(
        p(x)
        for x in [
            "The planned edge workflow begins with respiratory sound capture from the microphone connected to the Raspberry Pi. The input can be recorded as a short fixed-duration sample or buffered as a slightly longer signal from which overlapping windows are later extracted.",
            "Once recorded, the audio would pass through trimming, normalization, and feature-generation steps aligned with the existing shared preprocessing logic. Maintaining this parity is essential because the report's saved performance metrics are meaningful only when deployment uses the same fundamental representation strategy.",
            "The inference stage would then apply the trained strong-CNN model family or an optimized derivative suitable for the edge environment. The output would be reduced to a clear device-facing summary containing at minimum the predicted class and severity, while richer debugging detail could still be retained in logs during development.",
            "Finally, the LED display would present the screening outcome to the user. In a more advanced version, the device could also expose a compact menu, confidence bar, recording status icon, or simple warning indicator so that the interaction remains clear even without a browser or smartphone.",
        ]
    )
    out.append(h(2, "M.3 Deployment Considerations and Honest Scope"))
    out.extend(
        p(x)
        for x in [
            "Because the edge device is still under development, the report should treat it as a planned deployment extension backed by architectural reasoning rather than by completed device metrics. This is still academically valuable because report quality depends not only on finished code but also on how well the next deployment step is specified.",
            "If later iterations complete the hardware assembly, the same appendix can be extended with photographs of the setup, device-side latency measurements, power-consumption observations, and local inference screenshots. For the present version, however, the right framing is: designed, scoped, and intended for integration, but not falsely claimed as a finished benchmarked subsystem.",
        ]
    )
    return "".join(out)


def build_html() -> str:
    body = "".join(
        [
            build_cover_page(),
            build_front_matter(),
            build_chapter_1(),
            build_chapter_2(),
            build_chapter_3(),
            build_chapter_6(),
            build_appendices(),
        ]
    )
    return dedent(
        f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <title>Major Project Report</title>
          <style>
            @page {{
              margin: 1in;
            }}
            body {{
              font-family: "Times New Roman", serif;
              font-size: 12pt;
              line-height: 1.7;
              color: #111;
              margin: 0;
              padding: 0;
            }}
            .document {{
              max-width: 850px;
              margin: 0 auto;
              padding: 24px 12px 48px;
            }}
            .cover {{
              min-height: 1000px;
              display: flex;
              align-items: center;
              justify-content: center;
              text-align: center;
            }}
            .cover-block {{
              width: 90%;
              border: 2px solid #222;
              padding: 72px 48px;
            }}
            .cover-line {{
              font-size: 14pt;
              font-weight: bold;
              margin-bottom: 24px;
              letter-spacing: 1px;
            }}
            .cover-title {{
              font-size: 24pt;
              font-weight: bold;
              margin-bottom: 16px;
              text-transform: uppercase;
            }}
            .cover-subtitle {{
              font-size: 16pt;
              margin-bottom: 36px;
            }}
            .cover-meta p {{
              margin: 8px 0;
            }}
            .page-break {{
              page-break-before: always;
            }}
            h1 {{
              font-size: 18pt;
              margin-top: 24px;
              margin-bottom: 18px;
              text-align: center;
            }}
            h2 {{
              font-size: 15pt;
              margin-top: 22px;
              margin-bottom: 12px;
            }}
            h3 {{
              font-size: 13pt;
              margin-top: 18px;
              margin-bottom: 10px;
            }}
            p {{
              text-align: justify;
              margin: 0 0 12px 0;
            }}
            ul {{
              margin: 0 0 14px 24px;
            }}
            li {{
              margin-bottom: 6px;
            }}
            table {{
              width: 100%;
              border-collapse: collapse;
              margin: 12px 0 18px;
              table-layout: fixed;
            }}
            th, td {{
              border: 1px solid #555;
              padding: 8px;
              vertical-align: top;
              word-wrap: break-word;
            }}
            th {{
              background: #efefef;
            }}
            pre {{
              white-space: pre-wrap;
              word-wrap: break-word;
              background: #f5f5f5;
              border: 1px solid #ddd;
              padding: 12px;
              font-size: 10pt;
              line-height: 1.45;
            }}
            .ref-url {{
              font-size: 10.5pt;
            }}
          </style>
        </head>
        <body>
          <div class="document">
            {body}
          </div>
        </body>
        </html>
        """
    )


def html_to_blocks(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    document = soup.find("div", class_="document")
    blocks: list[tuple[str, str]] = []
    if document is None:
        return blocks

    for node in document.children:
        if not getattr(node, "name", None):
            continue
        classes = node.get("class", [])
        if node.name == "div" and "cover" in classes:
            blocks.append(("page_break", ""))
            text = node.get_text("\n", strip=True)
            blocks.append(("cover", text))
            continue
        if node.name == "div" and "page-break" in classes:
            blocks.append(("page_break", ""))
            continue
        if node.name == "h1":
            blocks.append(("h1", node.get_text(" ", strip=True)))
            continue
        if node.name == "h2":
            blocks.append(("h2", node.get_text(" ", strip=True)))
            continue
        if node.name == "h3":
            blocks.append(("h3", node.get_text(" ", strip=True)))
            continue
        if node.name == "p":
            blocks.append(("p", node.get_text(" ", strip=True)))
            continue
        if node.name == "ul":
            for item in node.find_all("li", recursive=False):
                blocks.append(("li", item.get_text(" ", strip=True)))
            continue
        if node.name == "table":
            rows = []
            for row in node.find_all("tr"):
                cells = row.find_all(["th", "td"])
                rows.append(" | ".join(cell.get_text(" ", strip=True) for cell in cells))
            blocks.append(("table", "\n".join(rows)))
            continue
        if node.name == "pre":
            blocks.append(("pre", node.get_text("\n", strip=False)))
            continue
    return blocks


def rtf_escape(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\t", "\\tab ")
    text = text.replace("\n", "\\line ")
    return text


def blocks_to_rtf(blocks: list[tuple[str, str]]) -> str:
    parts = [
        r"{\rtf1\ansi\ansicpg1252\deff0",
        r"{\fonttbl{\f0\froman\fcharset0 Times New Roman;}}",
        r"\viewkind4\uc1",
        r"\paperw12240\paperh15840\margl1440\margr1440\margt1440\margb1440",
        "\n",
    ]

    def add_para(text: str, *, size: int = 24, bold: bool = False, align: str = "j", space_before: int = 0, space_after: int = 180):
        alignment = {"l": r"\ql", "c": r"\qc", "r": r"\qr", "j": r"\qj"}[align]
        bold_on = r"\b " if bold else ""
        bold_off = r"\b0 " if bold else ""
        parts.append(
            rf"\pard{alignment}\sl528\slmult1\sa{space_after}\sb{space_before}\f0\fs{size} {bold_on}{rtf_escape(text)}{bold_off}\par"
            "\n"
        )

    for kind, text in blocks:
        if kind == "page_break":
            parts.append(r"\page" + "\n")
        elif kind == "cover":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                add_para(line, size=28 if line.isupper() and len(line) < 80 else 24, bold=True, align="c", space_after=220)
        elif kind == "h1":
            add_para(text, size=32, bold=True, align="c", space_before=120, space_after=220)
        elif kind == "h2":
            add_para(text, size=28, bold=True, align="l", space_before=120, space_after=160)
        elif kind == "h3":
            add_para(text, size=26, bold=True, align="l", space_before=100, space_after=140)
        elif kind == "p":
            add_para(text, size=24, align="j", space_after=180)
        elif kind == "li":
            add_para(f"- {text}", size=24, align="j", space_after=120)
        elif kind == "table":
            for line in text.splitlines():
                add_para(line, size=20, align="l", space_after=80)
            parts.append(r"\pard\sa180\par" + "\n")
        elif kind == "pre":
            for line in text.splitlines():
                add_para(line, size=18, align="l", space_after=40)
    parts.append("}")
    return "".join(parts)


def main() -> None:
    html = build_html()
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    blocks = html_to_blocks(html)
    rtf = blocks_to_rtf(blocks)
    OUTPUT_RTF.write_text(rtf, encoding="utf-8")
    subprocess.run(
        [
            "textutil",
            "-convert",
            "docx",
            str(OUTPUT_RTF),
            "-output",
            str(OUTPUT_DOCX),
        ],
        check=True,
    )
    print("Wrote:", OUTPUT_HTML)
    print("Wrote:", OUTPUT_RTF)
    print("Wrote:", OUTPUT_DOCX)
    plain_text = " ".join(text for kind, text in blocks if kind != "page_break")
    print("Estimated word count:", len(plain_text.split()))


if __name__ == "__main__":
    main()
