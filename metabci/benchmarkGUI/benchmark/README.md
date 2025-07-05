# MetaBenchmark-GUI

## Welcome! 
MetaBenchmark-GUI is an open-source platform for the open-source brain computer interface framework MetaBCI. The project focuses on multi module integration and visual interaction, unifying algorithm evaluation benchmarks, and improving the system integration, human-computer interaction efficiency, and architecture standardization of the MetaBCI platform. MetaBenchmark-GUI has 4 main innovation point:
* GUI platform: New GUI visualization interface integrating Brainstim, Brainda, and Brainflow，making it convenient to call various functions of MetaBCI architecture's three sub platforms on the GUI visualization platform, realizing the full process visualization of "data processing algorithm calling device interaction".
* Benchmark platform: Based on the Benchopt platform, build an algorithm evaluation system that covers performance metrics for multiple algorithms under three different brain computer interaction paradigms, and provide custom algorithm interfaces for users to test..
* Dataset analysis and processing platform:. 
* Online simulation testing platform:

This is the first release of MetaBenchmark-GUI, our team will continue to maintain the repository. If you need the handbook of this repository, please contact us by sending email to TBC_TJU_2022@163.com with the following information:
* Name of your teamleader
* Name of your university(or organization)

We will send you a copy of the handbook as soon as we receive your information.

## Content

- [MetaBenchmark-GUI](#metabenchmark-gui)
  - [Welcome!](#welcome)
  - [Content](#content)
  - [What are we doing?](#what-are-we-doing)
    - [The problem](#the-problem)
    - [The solution](#the-solution)
  - [Features](#features)
  - [Installation](#installation)
  - [Who are we?](#who-are-we)
  - [What do we need?](#what-do-we-need)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)
  - [Acknowledgements](#acknowledgements)

## What are we doing?

### The problem

* The existing toolchain is scattered
* Lack of algorithm evaluation system
* The workflow for analyzing and processing existing datasets is complex
* Lack of online simulation testing function

If someone wants to use MetaBCI for related functions, they need to search for program code scattered across various sub platforms, which is cumbersome and not intuitive to operate.At the same time, there is no way to quickly and intuitively compare the advantages and disadvantages of different algorithms on different datasets.

### The solution

The MetaBenchmark-GUI will:

* Allow users to load the data easily without knowing the details
* Provide flexible hook functions to control the preprocessing flow
* Provide the latest decoding algorithms
* Provide the experiment UI for different paradigms (e.g. MI, P300 and SSVEP)
* Provide the online data acquiring pipeline.
* Allow users to bring their pre-trained models to the online decoding pipeline.

The goal of the Meta-BCI is to make researchers focus on improving their own BCI algorithms and performing their experiments without wasting too much time on preliminary preparations.

## Features

* Improvements to MetaBCI
   - New GUI visualization interface integrating Brainstim, Brainda, and Brainflow
   - Add BenchMark algorithm benchmark testing function and its GUI
   - New dataset analysis and processing workflow and its GUI
   - Add simulation online testing function and its GUI
   - other small changes

* Supported Datasets
   - MI Datasets
     - AlexMI
     - BNCI2014001
     - PhysionetMI
     - MunichMI
     - Schirrmeister2017
     - Weibo2014
   - P300 Datasets
     - Cattan_P300
   - SSVEP Datasets
     - Nakanishi2015
     - Wang2016
     - BETA

* Supported reservation algorithms
   - SSVEP algorithms
     - ECCA、FBECCA、SCCA、FBSCCA、ItCCA、FBItCCA、TtCCA、FBTtCCA、MsetCCA、FBMsetCCA、MsetCCAR、FBMsetCCAR、TDCA、FBTDCA、TRCA、FBTRCA、TRCAR、FBTRCAR
   - P300 algorithms
     - DCPM、SKLDA、LDA、STDA
   - MI algorithms
     - CSP、 FBCSP、 MultiCSP、 FBMultiCSP、 DSP、 FBDSP、 SSCOR、 FBSSCOR

## Installation

1. Clone the repo
   ```sh
   git clone 
   ```
2. Change to the project directory
   ```sh
   cd MetaBenchmark-GUI
   ```
3. Create an environment and activate it
   ```sh
  conda create -n MetaBenchmark-GUI python=3.11.9
  conda activate MetaBenchmark-GUI
   ```
4. Install all requirements
   ```sh
   pip install -r requirements.txt 
   ```
5. Install brainda package with the editable mode
   ```sh
   pip install -e .
   ```
6. Install the latest version of Benchopt
  ```sh
  pip install -U benchopt
  # or latest development version
  pip install git+https://github.com/benchopt/benchopt.git
  ```
## Who are we?

The MetaBenchmark-GUI project is carried out by researchers from 
- Academy of Medical Engineering and Translational Medicine, Tianjin University, China


## What do we need?

**You**! In whatever way you can help.

We need expertise in programming, user experience, software sustainability, documentation and technical writing and project management.

We'd love your feedback along the way.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. **Any contributions you make are greatly appreciated**. Especially welcome to submit BCI algorithms.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v2.0 License. See `LICENSE` for more information.

## Contact

Email: 
Email: he_jy02@163.com

## Acknowledgements
- [MNE](https://github.com/mne-tools/mne-python)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [pyRiemann](https://github.com/alexandrebarachant/pyRiemann)
- [TRCA/eTRCA](https://github.com/mnakanishi/TRCA-SSVEP)
- [EEGNet](https://github.com/vlawhern/arl-eegmodels)
- [RPA](https://github.com/plcrodrigues/RPA)
- [MEKT](https://github.com/chamwen/MEKT)
