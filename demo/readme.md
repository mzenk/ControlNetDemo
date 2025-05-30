# ControlNet Demo

In order to start a demo on your local or any remote machine follow these steps:

1. Clone ControlNetDemo repository and create a virtual environment.
    ```
    $ git clone https://github.com/mzenk/ControlNetDemo.git
    $ cd ControlNetDemo
    $ conda env create -f environment.yaml
    $ conda activate control
    ```
2. Download a [model](https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth) from Huggingface. In particular put control_sd15_cannyp.pth (5.71GB!) in the ./models folder.
3. Run the demo:
    ```
    $ cd ControlNetDemo/demo
    $ python awesomedemo.py
    ```