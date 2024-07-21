# AudioFlamingo Inference and API Hosting

This repository is specifically for AudioFlamingo inference and API hosting.

## About AudioFlamingo

AudioFlamingo is a novel audio language model with few-shot learning and dialogue abilities. It represents a significant advancement in the field of audio processing and natural language interaction.

## Repository Purpose

The primary focus of this repository is to provide:

1. Inference capabilities for the AudioFlamingo model
2. API hosting solutions for easy integration into other applications

## Getting Started

Follow these steps to set up and use the AudioFlamingo API:

1. Clone this repository:
   ```
   git clone https://github.com/Sherry/flamingo-inference.git
   cd flamingo-inference
   ```

2. Install the required dependencies:
   ```
   python -m venv env python==3.8
   pip install -r requirements.txt
   ```

3. Start the API server:
   ```
   python inference/inference_endpoint.py
   ```

4. Use the API client to interact with the model:
   ```
   python inference/client.py
   ```

   This script provides examples of how to send requests to the API and process the responses.

5. Customize the `client.py` script or create your own client to integrate AudioFlamingo into your applications.

## License

The code in this repo is under MIT license (see LICENSE).

## References

If you use AudioFlamingo in your research or applications, please cite the original paper:

```bibtex
@article{kong2024audio,
  title={Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities},
  author={Kong, Zhifeng and Goel, Arushi and Badlani, Rohan and Ping, Wei and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2402.01831},
  year={2024}
}
```

Additional references:

```bibtex
@inproceedings{laionclap2023,
  title = {Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation},
  author = {Wu*, Yusong and Chen*, Ke and Zhang*, Tianyu and Hui*, Yuchen and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2023}
}

@inproceedings{htsatke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP},
  year = {2022}
}

@inproceedings{CLAP2022,
  title={Clap learning audio concepts from natural language supervision},
  author={Elizalde, Benjamin and Deshmukh, Soham and Al Ismail, Mahmoud and Wang, Huaming},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}

@misc{CLAP2023,
  title={Natural Language Supervision for General-Purpose Audio Representations}, 
  author={Benjamin Elizalde and Soham Deshmukh and Huaming Wang},
  year={2023},
  eprint={2309.05767},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2309.05767}
}
```