# hls-rnn-btag
Training RNN's for b-tagging for hls4ml

## Image location
```
gitlab-registry.cern.ch/rateixei/tfqkeras/tfqkeras:dylan_rnn_mastermerge
```

With docker, you can do
```
docker run -it gitlab-registry.cern.ch/rateixei/tfqkeras/tfqkeras:dylan_rnn_mastermerge
```

With singularity,
```
singularity shell docker://gitlab-registry.cern.ch/rateixei/tfqkeras/tfqkeras:dylan_rnn_mastermerge
```

This includes the keras-RNN branch from Dylan's hls4ml fork, plus all needed dependencies (including TensorFlow 2.4.1).
I can't make the C compilation work with Vivado, but the C-synthesis works. Need to check what's missing in the compiler side.

## Dataset location

Dataset is based on 7 TeV ttbar MC of CMS Open Data. You can download it here:
```https://cernbox.cern.ch/index.php/s/dYrWPhWQFbAgjh1 - pwd: hls-btag```
