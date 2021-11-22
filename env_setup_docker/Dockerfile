FROM tensorflow/tensorflow:latest

## ensure locale is set during build
ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# install git
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y g++-multilib

RUN apt-get install -y libc6-dev

RUN pip install --upgrade pip

RUN pip install argparse && \
    pip install nodejs && \
    pip install keras && \
    pip install awkward && \
    pip install uproot && \
    pip install jupyter && \
    pip install jupyterhub && \
    pip install jupyterlab && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install hep_ml && \
    pip install pandas && \
    pip install sklearn && \
    pip install tables && \
    pip install "dask[complete]" && \
    pip install papermill && \
    pip install pydot && \
    pip install Pillow && \
    pip install pyparser && \
    pip install pyparsing && \
    pip install pytest && \
    pip install tensorflow-model-optimization && \
    pip install networkx && \
    pip install prompt_toolkit==3.0.18 && \
    pip install keras-tuner && \
    pip install scikit-learn && \
    pip install tqdm && \
    pip install -e git://github.com/google/qkeras.git@v0.9.0#egg=qkeras && \
    pip install tensorflow_probability && \
    pip install h5py && \
    pip install pyyaml && \
    pip install protobuf==3.11.3 && \
    pip install onnx && \
    pip install -e git://github.com/rateixei/hls4ml.git@keras-rnn#egg=hls4ml
    

RUN apt-get install -y debconf-utils && \
    echo "krb5-config krb5-config/add_servers_realm string CERN.CH" | debconf-set-selections && \
    echo "krb5-config krb5-config/default_realm string CERN.CH" | debconf-set-selections && \
    apt-get install -y krb5-user && \
    apt-get install -y vim less screen graphviz python3-tk wget && \
    apt-get install -y jq tree hdf5-tools bash-completion

    
## run jupyter notebook by default unless a command is specified
## CMD ["jupyter", "notebook", "--port", "33333", "--no-browser"]

