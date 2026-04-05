"""
Generate DL Pipeline PDF Report
================================
Creates a comprehensive PDF report of the BMS Predictive Maintenance DL pipeline.
"""

import os
from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "DL_Pipeline_Report.pdf")


class DLReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "BMS Predictive Maintenance - DL Pipeline Report", align="C")
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, 15, 200, 15)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(20, 80, 60)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 80, 60)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 100, 80)
        self.ln(3)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.ln(2)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text, fill=True)
        self.ln(2)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(20, 80, 60)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        for j, row in enumerate(rows):
            self.set_fill_color(245, 250, 248) if j % 2 == 0 else self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)

    def add_image_if_exists(self, path, w=170):
        if os.path.exists(path):
            self.image(path, x=20, w=w)
            self.ln(5)


def build_report():
    pdf = DLReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ============================================================
    # COVER PAGE
    # ============================================================
    pdf.add_page()
    pdf.ln(50)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(20, 80, 60)
    pdf.cell(0, 15, "BMS Predictive Maintenance", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(40, 120, 90)
    pdf.cell(0, 12, "Deep Learning Pipeline Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_draw_color(20, 80, 60)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "SOH Estimation Using Deep Learning", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Battery A (5s1p, LG 18650HG2 NMC)", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "5 DL Models x 2 Feature Strategies = 10 Pipelines", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "LSTM | GRU | 1D-CNN | MLP | Transformer", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Birol - BMS Predictive Maintenance Project", align="C", new_x="LMARGIN", new_y="NEXT")

    # ============================================================
    # TABLE OF CONTENTS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(40, 40, 40)
    toc = [
        ("1.", "Project Overview & ML vs DL", 3),
        ("2.", "Data & Pipeline Steps", 4),
        ("3.", "Key Difference: Sequences & Scaling", 6),
        ("4.", "DL Models Explained", 8),
        ("5.", "Results & Rankings", 12),
        ("6.", "Key Findings & Analysis", 14),
        ("7.", "ML vs DL Final Comparison", 17),
        ("8.", "Limitations & Caveats", 18),
        ("9.", "Project Structure", 19),
        ("10.", "Next Steps", 20),
    ]
    for num, title, page in toc:
        pdf.cell(10, 8, num)
        pdf.cell(140, 8, title)
        pdf.cell(20, 8, str(page), align="R")
        pdf.ln()

    # ============================================================
    # 1. PROJECT OVERVIEW
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("1. Project Overview")

    pdf.section_title("Same Problem, Different Approach")
    pdf.body_text(
        "This is the same SOH estimation problem as the ML pipeline: predict battery State of Health "
        "from operational data (voltage, current, temperature, SOC). The difference is the models used - "
        "instead of tree-based algorithms (Random Forest, XGBoost), we use neural networks that can "
        "learn temporal patterns from sequences of battery cycles."
    )

    pdf.section_title("ML vs DL: Key Differences")
    pdf.add_table(
        ["Aspect", "ML (1-ML)", "DL (2-DL)"],
        [
            ["Models", "RF, XGBoost, LightGBM, etc.", "LSTM, GRU, CNN, MLP, Transformer"],
            ["Trend Features", "Manually computed", "Not needed (learned from sequence)"],
            ["Feature Count", "~200 (with trends) or 30", "~85 (no trends) or 30"],
            ["Feature Scaling", "Not required", "Required (StandardScaler)"],
            ["Input Shape", "2D: (samples, features)", "3D: (samples, 5, features)"],
            ["Temporal Context", "None (each row independent)", "Last 5 cycles together"],
            ["Training", "Single fit (seconds)", "Epochs + batches (minutes)"],
            ["Best Result", "ExtraTrees: MAE=0.064%", "GRU: MAE=0.249%"],
        ],
        [35, 70, 75]
    )

    # ============================================================
    # 2. PIPELINE STEPS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("2. Data & Pipeline Steps")

    pdf.section_title("Data (Same as ML)")
    pdf.add_table(
        ["Property", "Value"],
        [
            ["Source", "recording_senaryo_1_Battery_A_N100.mat"],
            ["Battery", "5s1p LG 18650HG2 NMC"],
            ["Sampling", "1 Hz, ~2.3M rows, ~100 cycles"],
            ["SOH Range", "100% -> 80%"],
        ],
        [60, 120]
    )

    pdf.section_title("Pipeline Overview (9 Steps)")
    pdf.add_table(
        ["Step", "Description", "Same as ML?"],
        [
            ["1", "Raw signal visualization (EDA)", "Yes"],
            ["2", "mV -> V conversion", "Yes"],
            ["3", "Cycle boundary detection", "Yes"],
            ["4", "Feature extraction (~85 features)", "Yes"],
            ["5", "Correlation analysis", "Yes"],
            ["6", "Feature selection (All or Top 30)", "Yes"],
            ["7", "Feature scaling + sequence prep", "NEW (DL only)"],
            ["8", "Model training (baseline)", "Different model"],
            ["9", "Hyperparameter tuning", "Different params"],
        ],
        [15, 110, 55]
    )

    pdf.body_text(
        "Steps 1-4 are identical to the ML pipeline - same .mat file reading, same cycle detection, "
        "same feature extraction. The key difference starts at Step 5: we do NOT compute trend features "
        "(rolling mean, delta, rate of change). DL sequence models learn these patterns automatically."
    )

    # Step 7 detail
    pdf.add_page()
    pdf.section_title("Step 7: Feature Scaling + Sequence Preparation (NEW)")

    pdf.subsection_title("Why Scaling is Required for DL")
    pdf.body_text(
        "Tree-based ML models split data at threshold points - the scale doesn't matter. "
        "Neural networks multiply inputs by weights and sum them. If voltage is 3.5V and "
        "temperature is 30C, the temperature will dominate just because it's a bigger number. "
        "StandardScaler normalizes each feature to mean=0, std=1, giving all features equal influence."
    )

    pdf.subsection_title("Sequence Preparation (Sliding Window)")
    pdf.body_text(
        "For sequence models (LSTM, GRU, CNN, Transformer), we create a sliding window of the last "
        "5 cycles for each prediction. Early cycles with insufficient history are zero-padded."
    )
    pdf.code_block(
        "Cycle 1:  [pad, pad, pad, pad, cycle1]   -> SOH_1\n"
        "Cycle 3:  [pad, pad, cycle1, cycle2, c3]  -> SOH_3\n"
        "Cycle 5:  [c1, c2, c3, c4, c5]            -> SOH_5\n"
        "Cycle 50: [c46, c47, c48, c49, c50]        -> SOH_50\n"
        "\n"
        "Shape: (n_samples, 5, n_features) -- 3D tensor"
    )
    pdf.body_text(
        "MLP is the exception - it uses flat input (no sequences), same as ML models but "
        "with a neural network. This serves as the DL baseline."
    )

    # ============================================================
    # 4. DL MODELS EXPLAINED
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("3. DL Models Explained")

    # ANN vs CNN vs RNN
    pdf.section_title("Three Families of Neural Networks")
    pdf.add_table(
        ["Family", "Input", "Memory", "Best For", "Our Model"],
        [
            ["ANN", "Fixed vector", "None", "Tabular data", "MLP"],
            ["CNN", "Grid/signal", "None", "Local patterns", "1D-CNN"],
            ["RNN", "Sequence", "Yes", "Time series", "LSTM, GRU"],
            ["Attention", "Sequence", "Via attention", "Long sequences", "Transformer"],
        ],
        [28, 35, 25, 42, 40]
    )

    # LSTM
    pdf.section_title("LSTM (Long Short-Term Memory)")
    pdf.body_text(
        "Category: RNN (Recurrent Neural Network)\n\n"
        "LSTM processes cycles one by one, maintaining a memory cell with three gates:\n"
        "- Forget Gate: decides what to discard from memory\n"
        "- Input Gate: decides what new information to store\n"
        "- Output Gate: decides what to output\n\n"
        "Our architecture: Bidirectional LSTM (reads forward AND backward) with 2 layers. "
        "This captures patterns in both directions of the sequence."
    )
    pdf.add_table(
        ["Parameter", "Baseline", "Tuning Range"],
        [
            ["hidden_dim", "64", "32, 64, 128"],
            ["n_layers", "2", "1, 2, 3"],
            ["learning_rate", "0.001", "0.0005, 0.001, 0.005"],
            ["dropout", "0.2", "fixed"],
            ["seq_len", "5", "fixed"],
        ],
        [50, 40, 90]
    )

    # GRU
    pdf.section_title("GRU (Gated Recurrent Unit)")
    pdf.body_text(
        "Category: RNN (simplified LSTM)\n\n"
        "Same concept as LSTM but with only 2 gates (reset + update) instead of 3. "
        "This means fewer parameters, faster training, and often similar or better performance "
        "on small datasets. Think of GRU as 'LSTM lite'.\n\n"
        "Our GRU achieved the best DL result (MAE=0.249%) - its simplicity is an advantage "
        "with only 100 training cycles."
    )

    # CNN
    pdf.add_page()
    pdf.section_title("1D-CNN (1D Convolutional Neural Network)")
    pdf.body_text(
        "Category: CNN\n\n"
        "Applies learned filters (kernels) that slide across the cycle sequence. "
        "Two Conv1d layers detect local patterns (e.g., 'voltage spread increased for 3 consecutive "
        "cycles'), followed by global average pooling to produce a single feature vector.\n\n"
        "Unlike LSTM/GRU which process sequentially, CNN processes the entire sequence at once. "
        "However, with only 5 timesteps, there isn't much spatial structure for CNN to exploit - "
        "which explains its weaker performance."
    )
    pdf.add_table(
        ["Parameter", "Baseline", "Tuning Range"],
        [
            ["n_filters", "64", "32, 64, 128"],
            ["dropout", "0.2", "0.1, 0.2, 0.3"],
            ["learning_rate", "0.001", "0.0005, 0.001, 0.005"],
            ["kernel_size", "3", "fixed"],
        ],
        [50, 40, 90]
    )

    # MLP
    pdf.section_title("MLP (Multi-Layer Perceptron)")
    pdf.body_text(
        "Category: ANN (fully connected)\n\n"
        "The simplest DL model - no sequence processing. Each cycle is treated independently, "
        "just like ML models. Three hidden layers with BatchNorm and Dropout.\n\n"
        "Purpose: DL baseline. If sequence models don't beat MLP, temporal information isn't helping.\n\n"
        "Key finding: MLP Baseline was catastrophic (MAE=13.58%) because default hyperparameters "
        "were terrible. After tuning, MLP Tuned achieved MAE=0.787% - showing that DL is extremely "
        "sensitive to hyperparameter choices."
    )

    # Transformer
    pdf.section_title("Transformer (Encoder-Only)")
    pdf.body_text(
        "Category: Attention-based\n\n"
        "The architecture behind GPT, BERT, and ChatGPT - adapted for our SOH prediction task.\n\n"
        "Key concept: Self-Attention. Instead of processing cycles sequentially (LSTM) or with "
        "local filters (CNN), the Transformer lets each cycle 'attend' to all other cycles "
        "simultaneously. It learns which cycles are most relevant for prediction.\n\n"
        "Requires Positional Encoding because Transformer has no inherent notion of order - "
        "we must explicitly tell it which cycle is 1st, 2nd, etc.\n\n"
        "On our small dataset, Transformer performs moderately (MAE=0.441%). Its true power "
        "emerges with larger datasets and longer sequences."
    )
    pdf.add_table(
        ["Parameter", "Baseline", "Tuning Range"],
        [
            ["d_model", "64", "32, 64, 128"],
            ["nhead", "4", "fixed"],
            ["n_layers", "2", "1, 2, 3"],
            ["learning_rate", "0.001", "0.0005, 0.001, 0.005"],
        ],
        [50, 40, 90]
    )

    # ============================================================
    # 5. RESULTS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("4. Results & Rankings")

    pdf.section_title("All DL Models Ranked by Test MAE")

    results = [
        ["1", "GRU_Top30_Tuned", "0.249%", "0.299%", "0.9970", "0.598%"],
        ["2", "GRU_Tuned", "0.295%", "0.371%", "0.9954", "0.802%"],
        ["3", "GRU_Top30_BL", "0.308%", "0.380%", "0.9952", "0.795%"],
        ["4", "LSTM_Top30_Tuned", "0.410%", "0.577%", "0.9889", "--"],
        ["5", "Transformer_Tuned", "0.441%", "0.598%", "0.9881", "1.025%"],
        ["6", "GRU_BL", "0.519%", "0.687%", "0.9843", "3.790%"],
        ["7", "TF_Top30_Tuned", "0.531%", "0.828%", "0.9772", "0.884%"],
        ["8", "TF_Top30_BL", "0.610%", "0.908%", "0.9725", "0.625%"],
        ["9", "LSTM_Tuned", "0.638%", "0.751%", "0.9812", "3.350%"],
        ["10", "MLP_Tuned", "0.787%", "0.851%", "0.9759", "0.973%"],
        ["11", "LSTM_Top30_BL", "0.845%", "1.031%", "0.9645", "--"],
        ["12", "MLP_Top30_Tuned", "0.849%", "0.960%", "0.9693", "1.452%"],
        ["13", "TF_BL", "0.880%", "1.028%", "0.9648", "0.879%"],
        ["14", "CNN_Tuned", "0.923%", "1.146%", "0.9563", "1.719%"],
        ["15", "CNN_Top30_Tuned", "1.001%", "1.241%", "0.9487", "3.863%"],
        ["16", "LSTM_BL", "1.370%", "1.586%", "0.9162", "1.480%"],
        ["17", "CNN_Top30_BL", "1.680%", "1.988%", "0.8682", "7.310%"],
        ["18", "CNN_BL", "1.781%", "2.170%", "0.8431", "3.689%"],
        ["19", "MLP_BL", "13.58%", "14.05%", "-5.584", "47.42%"],
        ["20", "MLP_Top30_BL", "19.80%", "21.04%", "-13.76", "42.98%"],
    ]

    pdf.set_font("Helvetica", "", 8)
    pdf.add_table(
        ["#", "Model", "MAE", "RMSE", "R2", "CV MAE"],
        results,
        [8, 48, 22, 22, 22, 22]
    )

    # DL Comparison image
    img_path = os.path.join(SCRIPT_DIR, "DL_Comparison.png")
    pdf.add_image_if_exists(img_path, w=180)

    # ============================================================
    # 6. KEY FINDINGS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("5. Key Findings & Analysis")

    pdf.section_title("Finding 1: GRU is the Best DL Model (MAE = 0.249%)")
    pdf.body_text(
        "GRU Top30 Tuned achieved the lowest DL MAE. Its simpler architecture (2 gates vs LSTM's 3) "
        "makes it more parameter-efficient, which is crucial with only 100 training cycles. "
        "GRU proves that simpler models often win on small datasets."
    )

    pdf.section_title("Finding 2: GRU > LSTM on Small Data")
    pdf.add_table(
        ["Model", "Best MAE", "Parameters", "Why"],
        [
            ["GRU Top30", "0.249%", "Fewer (2 gates)", "More efficient"],
            ["LSTM Top30", "0.410%", "More (3 gates)", "Needs more data"],
        ],
        [40, 30, 45, 65]
    )

    pdf.section_title("Finding 3: Feature Selection Still Matters in DL")
    pdf.add_table(
        ["Model", "All Features", "Top 30", "Winner"],
        [
            ["GRU", "0.295%", "0.249%", "Top 30"],
            ["LSTM", "0.638%", "0.410%", "Top 30"],
            ["Transformer", "0.441%", "0.531%", "All"],
            ["CNN", "0.923%", "1.001%", "All"],
            ["MLP", "0.787%", "0.849%", "All"],
        ],
        [42, 38, 38, 38]
    )
    pdf.body_text(
        "RNN models (GRU, LSTM) benefit from Top 30 selection - extra features add noise that "
        "degrades sequential learning. CNN, MLP, and Transformer prefer more features. "
        "The lesson: 'DL learns everything automatically' is NOT true with 100 samples."
    )

    pdf.section_title("Finding 4: Hyperparameter Tuning is Critical")
    pdf.add_table(
        ["Model", "Baseline MAE", "Tuned MAE", "Improvement"],
        [
            ["MLP", "13.582%", "0.787%", "17x better!"],
            ["LSTM", "1.370%", "0.638%", "2.1x"],
            ["GRU", "0.519%", "0.295%", "1.8x"],
            ["CNN", "1.781%", "0.923%", "1.9x"],
            ["Transformer", "0.880%", "0.441%", "2.0x"],
        ],
        [40, 40, 40, 40]
    )
    pdf.body_text(
        "DL is far more sensitive to hyperparameters than ML. MLP Baseline had catastrophic R2=-5.58 "
        "(worse than random). After tuning: R2=0.976. In ML, baseline RF already gave R2=0.999."
    )

    pdf.add_page()
    pdf.section_title("Finding 5: CNN Underperforms on Short Sequences")
    pdf.body_text(
        "CNN (MAE ~0.92-1.78%) consistently underperforms RNNs and Transformer. 1D convolution "
        "filters are designed for local pattern detection in long sequences. With only 5 timesteps, "
        "there isn't enough spatial structure. CNN would likely improve with longer sequences (seq_len=20+)."
    )

    pdf.section_title("Finding 6: Transformer - Moderate on Small Data")
    pdf.body_text(
        "Transformer Tuned achieved MAE=0.441%, ranking 5th. Self-attention is powerful but "
        "parameter-heavy. With 100 cycles and seq_len=5, there isn't enough data to leverage "
        "Transformer's full potential. On larger datasets (1000+ cycles), Transformer typically "
        "outperforms RNNs."
    )

    pdf.section_title("Finding 7: MLP Baseline is a Cautionary Tale")
    pdf.body_text(
        "MLP Baseline: MAE=13.58%, R2=-5.58. This means the model's predictions were WORSE than "
        "simply predicting the mean SOH for every cycle. The random weight initialization, combined "
        "with suboptimal learning rate and architecture, led to complete failure.\n\n"
        "Lesson: DL models are NOT plug-and-play. They require careful hyperparameter selection. "
        "ML models (especially tree-based) are much more forgiving with default parameters."
    )

    # ============================================================
    # 7. ML vs DL
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("6. ML vs DL Final Comparison")

    pdf.add_table(
        ["", "Best ML", "Best DL"],
        [
            ["Model", "ExtraTrees Top30 BL", "GRU Top30 Tuned"],
            ["MAE", "0.064%", "0.249%"],
            ["RMSE", "0.079%", "0.299%"],
            ["R2", "0.9998", "0.9970"],
            ["Winner", "ML", "--"],
        ],
        [40, 70, 70]
    )

    pdf.body_text(
        "ML wins on this dataset. This is expected and well-documented in the literature:\n\n"
        "1. Tree-based models excel on small tabular data (<1000 samples)\n"
        "2. DL needs more data to learn complex patterns\n"
        "3. ML had manually computed trend features (~200 features) - domain knowledge advantage\n"
        "4. DL models are more sensitive to hyperparameters\n\n"
        "However, DL has unique advantages that don't show on this dataset:\n"
        "- Extrapolation: DL can potentially predict beyond training range (trees cannot)\n"
        "- Transfer learning: Pre-train on one battery, fine-tune on another\n"
        "- Scalability: DL improves with more data, ML plateaus"
    )

    pdf.section_title("When Would DL Win?")
    pdf.add_table(
        ["Data Size", "Expected Winner", "Why"],
        [
            ["<500 cycles", "ML", "Not enough data for DL"],
            ["500-5000", "Close / DL starts winning", "DL patterns emerge"],
            ["5000+", "DL", "DL fully leverages patterns"],
            ["Transfer learning", "DL", "Pre-trained models adapt fast"],
        ],
        [40, 55, 85]
    )

    # ============================================================
    # 8. LIMITATIONS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("7. Limitations & Caveats")

    pdf.section_title("100 Cycles is Very Limited for DL")
    pdf.body_text(
        "Deep learning typically needs thousands of samples. With 100 cycles, all DL models "
        "are data-starved. The results would likely be very different with 10 batteries x 100 "
        "cycles = 1000 training samples."
    )

    pdf.section_title("Feature Engineering Still Critical")
    pdf.body_text(
        "We still manually designed domain features (voltage spread, proxy IR, etc.) in Step 4. "
        "True 'end-to-end DL' would take raw 1 Hz signals directly - but with 100 cycles of "
        "23000-second signals, this is computationally infeasible and would overfit massively."
    )

    pdf.section_title("Interleaved MAE is Optimistic")
    pdf.body_text(
        "Same caveat as ML. GRU's interleaved MAE=0.249% but CV MAE=0.598%. Real-world "
        "performance is closer to the CV MAE. Fold 5 (chronological test) would be even higher."
    )

    pdf.section_title("Non-Deterministic Training")
    pdf.body_text(
        "DL training involves random weight initialization and stochastic gradient descent. "
        "Results vary slightly between runs. We use random seeds for reproducibility, but "
        "GPU operations can introduce minor variations."
    )

    # ============================================================
    # 9. PROJECT STRUCTURE
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("8. Project Structure")

    pdf.code_block(
        "2-DL/\n"
        "|-- LSTM/\n"
        "|   |-- LSTM.py, LSTM_Top30.py\n"
        "|   |-- Output_LSTM/, Output_LSTM_Top30/\n"
        "|-- GRU/\n"
        "|   |-- GRU.py, GRU_Top30.py\n"
        "|-- CNN/\n"
        "|   |-- CNN.py, CNN_Top30.py\n"
        "|-- MLP/\n"
        "|   |-- MLP.py, MLP_Top30.py\n"
        "|-- Transformer/\n"
        "|   |-- Transformer.py, Transformer_Top30.py\n"
        "|-- DL_Comparison.py\n"
        "|-- DL_Comparison.csv / .png\n"
        "|-- DL_Pipeline_Report.pdf (this document)\n"
        "|-- README.md"
    )

    pdf.section_title("Output Files Per Model")
    pdf.add_table(
        ["File", "Content"],
        [
            ["step[1-7]_*.png", "Pipeline step visualizations"],
            ["step8_*.png", "Training: metrics, prediction, residuals, loss curve"],
            ["step9_*.png", "Tuning: baseline vs tuned, tuned prediction"],
            ["model_*_baseline.pt", "Baseline PyTorch model weights"],
            ["model_*_tuned.pt", "Tuned PyTorch model weights"],
            ["scaler.joblib", "StandardScaler for inference"],
            ["feature_columns.txt", "Feature list for inference"],
            ["Result_*.csv", "Summary with overfitting checks + CV MAE"],
        ],
        [55, 125]
    )

    # ============================================================
    # 10. NEXT STEPS
    # ============================================================
    pdf.add_page()
    pdf.chapter_title("9. Next Steps")

    pdf.section_title("Battery B Inference")
    pdf.body_text(
        "Apply trained DL models to Battery B (unseen battery, ground truth SOH=88%). "
        "Same feature extraction pipeline, load saved scaler and model weights, predict SOH."
    )

    pdf.section_title("ML + DL Combined Comparison")
    pdf.body_text(
        "Create a unified comparison chart showing all ML (12 models) and DL (20 models) "
        "side by side. This gives the complete picture of model performance on this dataset."
    )

    pdf.section_title("Transfer Learning")
    pdf.body_text(
        "Pre-train DL models on Battery A, then fine-tune on Battery B with minimal data. "
        "This is where DL is expected to significantly outperform ML - leveraging learned "
        "degradation patterns from one battery to predict another."
    )

    pdf.section_title("Larger Datasets")
    pdf.body_text(
        "Combine multiple batteries (A + B + additional simulations) to create a dataset "
        "of 500+ cycles. This would allow DL models to fully leverage their capacity and "
        "likely surpass ML performance."
    )

    pdf.section_title("Ensemble Methods")
    pdf.body_text(
        "Combine the best ML model (ExtraTrees, MAE=0.064%) with the best DL model "
        "(GRU, MAE=0.249%) through prediction averaging or stacking. Ensemble methods "
        "often outperform any single model."
    )

    # Save
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved: {OUTPUT_PATH}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_report()
