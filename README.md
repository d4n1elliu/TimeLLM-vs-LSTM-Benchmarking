# TimeLLM-vs-LSTM-Benchmarking
This project provides a rigorous comparative analysis between Large Language Models (LLMs) adapted for temporal data and traditional Long Short-Term Memory (LSTM) networks. By evaluating performance across predictive accuracy, computational efficiency, and architectural robustness, this research identifies the trade-offs between "foundation model" approaches and domain-specific recurrent neural networks.

### 🧪 Core ObjectivesModel Implementation: 
- Development of a high-performance LSTM baseline and a contemporary TimeLLM-based prototype.
- Benchmarking: Assessing models on standard time-series datasets using metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE).
- Experimental Validation: Testing the "zero-shot" and "few-shot" capabilities of LLMs against the supervised learning requirements of LSTMs.

### 🏗️ Architectures Compared

| Feature | LSTM Prototype | LLM-based Prototype |
| :--- | :--- | :--- |
| **Type** | Recurrent Neural Network (RNN) | Transformer Foundation Model |
| **Strengths** | Low latency, efficient for small data | Long-range dependencies, zero-shot |
| **Data Handling** | Sequential hidden states | Temporal patches & embedding |
| **Training** | Fully supervised (from scratch) | Fine-tuning or Prompt-based |

### 📈 Key Research Findings

- Performance: The comparison highlights how LLMs leverage pre-trained linguistic patterns to understand cyclicality in data that traditional LSTMs may miss without extensive feature engineering.
- Robustness: Experimental results demonstrate that while LSTMs remain highly efficient for narrow, high-frequency tasks, the LLM approach offers superior adaptability across varying time horizons.
- Novelty: The implementation includes a mechanism to translate numerical time-series data into high-dimensional embeddings suitable for transformer attention heads.

### 🛠️ Tech Stack 
Languages: Python

Libraries: PyTorch/TensorFlow, NumPy, Pandas, Scikit-learn

Environment: Jupyter Notebooks (.ipynb)

Hardware Target: Optimized for CUDA-enabled GPUs to handle LLM inference overhead.
