# PyRIT Prompt Test Framework

A red teaming test framework based on the Microsoft library Python Risk Identification Tool for generative AI (PyRIT).
Different dataset of prompts are used to stress and probe a target LLM. PyRIT APIs are used to load and model the prompts, to handle the communication with the target LLM and to apply different sets of scorers to the LLM outputs. The results are collected in a report.

## 🔧 Setup (Windows)

```bash
git clone https://github.com/your-org/pyrit_test_framework.git
copy .env.example .env
cd pyrit_test_framework
REM Edit .env with the Azure extremes of the target LLM model
setup.bat
