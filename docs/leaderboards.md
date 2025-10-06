---
title: Leaderboards
---

# Updates
* **2025-09-01**: **[Rank-sum Leaderboard]** Segmentation results were updated (small changes based on improved performance for all models on the *ocelot* dataset only -> [Related commit](https://github.com/MICS-Lab/thunder/commit/5f6d6e7cdd6a1df5affed2dac47233f80ce5a205)). Segmentation and global rankings do no thus match exactly (a few small differences only) Table 4 from the current version of our [arXiv paper](https://arxiv.org/abs/2507.07860). The paper will be updated soon.
* **2025-09-30**: **[SPIDER Leaderboard]** Four SPIDER datasets have been integrated into thunder. Results associated to them have not been integrated into the rank-sum leaderboard (only datasets presented in our [arXiv paper](https://arxiv.org/abs/2507.07860) are aggregated in the rank-sum learderboard), but we have instead created a leaderboard dedicated to SPIDER datasets below.
* **2025-10-06**: **[Zero-shot VLM Classification Leaderboard]** A new zero-shot classification task was integrated into THUNDER. Results are presented in a dedicated leaderboard below.

# 🏆 Rank-sum Leaderboard

<div class="table-responsive-sm">
  <table id="ranksumTable" class="table table-hover table-bordered table-sm nowrap">
    <thead class="align-middle text-center">
      <tr>
        <th>Model</th>
        <th>Domain</th>
        <th>Type</th>
        <th>KNN &uarr;</th>
        <th>Lin. prob. &uarr;</th>
        <th>Few-shot &uarr;</th>
        <th>Seg.&uarr;</th>
        <th>Calib. &darr;</th>
        <th>Adv. attack &darr;</th>
        <th>Rank sum &darr;</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>HIBOU-B</td><td>Histopathology</td><td>VM</td><td>75.8 (10)</td><td>78.0 (14)</td><td>74.2 (6)</td><td>67.8 (10)</td><td>3.7 (2)</td><td>52.8 (14)</td><td>56 (7)</td></tr>
      <tr><td>HIBOU-L</td><td>Histopathology</td><td>VM</td><td>75.2 (12)</td><td>81.2 (7)</td><td>70.4 (12)</td><td>68.6 (6)</td><td>5.5 (18)</td><td>40.0 (5)</td><td>60 (8)</td></tr>
      <tr><td>H-OPTIMUS-0</td><td>Histopathology</td><td>VM</td><td>79.2 (5)</td><td>81.4 (5)</td><td>73.4 (7)</td><td>65.2 (13)</td><td>4.7 (13)</td><td>44.2 (9)</td><td>52 (6)</td></tr>
      <tr><td>H-OPTIMUS-1</td><td>Histopathology</td><td>VM</td><td>80.5 (3)</td><td>83.3 (2)</td><td>74.8 (4)</td><td>64.5 (15)</td><td>4.1 (4)</td><td>58.0 (17)</td><td>45 (5)</td></tr>
      <tr><td>MIDNIGHT</td><td>Histopathology</td><td>VM</td><td>78.2 (8)</td><td>82.9 (3)</td><td>70.6 (11)</td><td>68.8 (4)</td><td>3.2 (1)</td><td>36.3 (4)</td><td>31 (3)</td></tr>
      <tr><td>PHIKON</td><td>Histopathology</td><td>VM</td><td>72.8 (14)</td><td>78.4 (13)</td><td>72.2 (10)</td><td>68.0 (9)</td><td>6.4 (22)</td><td>34.4 (3)</td><td>71 (11)</td></tr>
      <tr><td>PHIKON2</td><td>Histopathology</td><td>VM</td><td>70.1 (15)</td><td>76.5 (15)</td><td>70.1 (13)</td><td>67.4 (12)</td><td>4.6 (11)</td><td>45.6 (11)</td><td>77 (12)</td></tr>
      <tr><td>UNI</td><td>Histopathology</td><td>VM</td><td>78.8 (6)</td><td>81.3 (6)</td><td>76.4 (2)</td><td>67.8 (11)</td><td>4.3 (7)</td><td>42.8 (7)</td><td>39 (4)</td></tr>
      <tr><td>UNI2-H</td><td>Histopathology</td><td>VM</td><td>81.7 (1)</td><td>83.9 (1)</td><td>78.4 (1)</td><td>69.0 (3)</td><td>4.5 (8)</td><td>34.3 (2)</td><td>16 (1)</td></tr>
      <tr><td>VIRCHOW</td><td>Histopathology</td><td>VM</td><td>74.2 (13)</td><td>80.2 (10)</td><td>68.5 (15)</td><td>69.2 (2)</td><td>5.5 (20)</td><td>41.0 (6)</td><td>66 (10)</td></tr>
      <tr><td>VIRCHOW2</td><td>Histopathology</td><td>VM</td><td>81.2 (2)</td><td>82.7 (4)</td><td>72.6 (9)</td><td>69.3 (1)</td><td>4.6 (10)</td><td>33.6 (1)</td><td>27 (2)</td></tr>
      <tr><td>CONCH</td><td>Histopathology</td><td>VLM</td><td>77.3 (9)</td><td>80.2 (11)</td><td>73.1 (8)</td><td>68.3 (7)</td><td>4.3 (6)</td><td>55.0 (15)</td><td>56 (7)</td></tr>
      <tr><td>CONCH&nbsp;1.5</td><td>Histopathology</td><td>VLM</td><td>78.6 (7)</td><td>80.8 (9)</td><td>74.6 (5)</td><td>68.8 (5)</td><td>4.9 (14)</td><td>75.3 (23)</td><td>63 (9)</td></tr>
      <tr><td>KEEP</td><td>Histopathology</td><td>VLM</td><td>79.7 (4)</td><td>81.1 (8)</td><td>75.8 (3)</td><td>68.0 (8)</td><td>4.7 (12)</td><td>44.7 (10)</td><td>45 (5)</td></tr>
      <tr><td>MUSK</td><td>Histopathology</td><td>VLM</td><td>75.6 (11)</td><td>79.0 (12)</td><td>70.0 (14)</td><td>65.1 (14)</td><td>4.5 (9)</td><td>69.3 (22)</td><td>82 (13)</td></tr>
      <tr><td>PLIP</td><td>Histopathology</td><td>VLM</td><td>67.8 (19)</td><td>71.0 (22)</td><td>63.4 (17)</td><td>58.5 (22)</td><td>4.9 (15)</td><td>56.9 (16)</td><td>111 (18)</td></tr>
      <tr><td>QUILTNET</td><td>Histopathology</td><td>VLM</td><td>68.3 (17)</td><td>71.0 (21)</td><td>65.7 (16)</td><td>58.9 (21)</td><td>7.0 (23)</td><td>52.7 (13)</td><td>111 (18)</td></tr>
      <tr><td>DINOv2-B</td><td>Natural-image</td><td>VM</td><td>67.9 (18)</td><td>74.8 (17)</td><td>61.0 (18)</td><td>59.8 (19)</td><td>5.5 (21)</td><td>65.8 (20)</td><td>113 (19)</td></tr>
      <tr><td>DINOv2-L</td><td>Natural-image</td><td>VM</td><td>69.6 (16)</td><td>75.3 (16)</td><td>59.2 (19)</td><td>59.6 (20)</td><td>5.3 (17)</td><td>64.5 (19)</td><td>107 (17)</td></tr>
      <tr><td>ViT-B/16</td><td>Natural-image</td><td>VM</td><td>64.4 (21)</td><td>71.9 (19)</td><td>57.8 (21)</td><td>61.0 (17)</td><td>3.9 (3)</td><td>46.8 (12)</td><td>93 (14)</td></tr>
      <tr><td>ViT-L/16</td><td>Natural-image</td><td>VM</td><td>67.5 (20)</td><td>72.8 (18)</td><td>56.5 (22)</td><td>63.1 (16)</td><td>5.0 (16)</td><td>44.1 (8)</td><td>100 (15)</td></tr>
      <tr><td>CLIP-B/32</td><td>Natural-image</td><td>VLM</td><td>61.9 (23)</td><td>65.8 (23)</td><td>53.3 (23)</td><td>56.0 (23)</td><td>5.5 (19)</td><td>60.4 (18)</td><td>129 (23)</td></tr>
      <tr><td>CLIP-L/14</td><td>Natural-image</td><td>VLM</td><td>64.2 (22)</td><td>71.3 (20)</td><td>58.2 (20)</td><td>60.8 (18)</td><td>4.2 (5)</td><td>67.8 (21)</td><td>106 (16)</td></tr>
    </tbody>
  </table>
</div>

---

# 🏆 SPIDER Leaderboard

F1-score on test sets of SPIDER datasets and average across datasets for the *knn* and *linear probing* tasks. The considered datasets are:

* *Br*: [SPIDER-Breast](https://huggingface.co/datasets/histai/SPIDER-breast)
* *Co*: [SPIDER-Colorectal](https://huggingface.co/datasets/histai/SPIDER-colorectal)
* *Sk*: [SPIDER-skin](https://huggingface.co/datasets/histai/SPIDER-skin)
* *Th*: [SPIDER-thorax](https://huggingface.co/datasets/histai/SPIDER-thorax)

<div class="table-responsive-sm">
    <table id="spiderTable" class="table table-hover table-bordered table-sm nowrap">
        <thead class="align-middle text-center">
          <tr>
            <th rowspan="2">Model</th>
            <th rowspan="2">Domain</th>
            <th rowspan="2">Type</th>
            <th colspan="5">KNN &uarr;</th>
            <th colspan="5">Linear probing &uarr;</th>
          </tr>
          <tr>
            <th>Br</th><th>Co</th><th>Sk</th><th>Th</th><th>Avg</th>
            <th>Br</th><th>Co</th><th>Sk</th><th>Th</th><th>Avg</th>
          </tr>
      </thead>
        <tbody>
          <tr><td>HIBOU-B</td><td>Histopathology</td><td>VM</td><td>83.3 (2)</td><td>88.1 (1)</td><td>87.7 (7)</td><td>93.4 (3)</td><td>88.1 (5)</td><td>86.6 (5)</td><td>90.7 (2)</td><td>91.1 (8)</td><td>94.5 (4)</td><td>90.7 (5)</td></tr>
          <tr><td>HIBOU-L</td><td>Histopathology</td><td>VM</td><td>83.6 (1)</td><td>88.1 (2)</td><td>90.7 (2)</td><td>93.5 (2)</td><td>89.0 (1)</td><td>88.0 (1)</td><td>89.8 (8)</td><td>93.3 (1)</td><td>94.1 (7)</td><td>91.3 (1)</td></tr>
          <tr><td>H-OPTIMUS-0</td><td>Histopathology</td><td>VM</td><td>81.7 (6)</td><td>87.8 (7)</td><td>89.3 (4)</td><td>93.8 (1)</td><td>88.2 (4)</td><td>87.2 (3)</td><td>89.9 (7)</td><td>91.9 (5)</td><td>94.4 (6)</td><td>90.8 (4)</td></tr>
          <tr><td>H-OPTIMUS-1</td><td>Histopathology</td><td>VM</td><td>83.0 (3)</td><td>87.8 (5)</td><td>91.1 (1)</td><td>91.5 (9)</td><td>88.4 (2)</td><td>86.1 (8)</td><td>90.3 (5)</td><td>92.3 (3)</td><td>93.6 (11)</td><td>90.6 (6)</td></tr>
          <tr><td>MIDNIGHT</td><td>Histopathology</td><td>VM</td><td>77.1 (13)</td><td>84.9 (13)</td><td>85.7 (10)</td><td>92.7 (5)</td><td>85.1 (11)</td><td>86.1 (7)</td><td>89.6 (11)</td><td>91.0 (9)</td><td>94.4 (5)</td><td>90.3 (9)</td></tr>
          <tr><td>PHIKON</td><td>Histopathology</td><td>VM</td><td>78.9 (11)</td><td>85.1 (12)</td><td>83.2 (13)</td><td>89.7 (15)</td><td>84.3 (12)</td><td>84.9 (12)</td><td>88.5 (12)</td><td>87.9 (11)</td><td>92.4 (13)</td><td>88.4 (12)</td></tr>
          <tr><td>PHIKON2</td><td>Histopathology</td><td>VM</td><td>80.2 (8)</td><td>86.5 (10)</td><td>83.3 (11)</td><td>91.4 (11)</td><td>85.3 (10)</td><td>86.0 (9)</td><td>89.7 (10)</td><td>87.2 (14)</td><td>94.7 (3)</td><td>89.4 (11)</td></tr>
          <tr><td>UNI</td><td>Histopathology</td><td>VM</td><td>81.3 (7)</td><td>88.0 (4)</td><td>87.2 (8)</td><td>91.2 (12)</td><td>86.9 (8)</td><td>85.7 (10)</td><td>90.4 (4)</td><td>91.2 (7)</td><td>93.9 (8)</td><td>90.3 (8)</td></tr>
          <tr><td>UNI2-H</td><td>Histopathology</td><td>VM</td><td>82.6 (4)</td><td>87.1 (9)</td><td>90.5 (3)</td><td>92.5 (7)</td><td>88.2 (3)</td><td>86.7 (4)</td><td>90.5 (3)</td><td>92.5 (2)</td><td>95.1 (1)</td><td>91.2 (2)</td></tr>
          <tr><td>VIRCHOW</td><td>Histopathology</td><td>VM</td><td>79.3 (10)</td><td>87.8 (6)</td><td>88.8 (6)</td><td>92.3 (8)</td><td>87.0 (7)</td><td>86.2 (6)</td><td>90.2 (6)</td><td>91.3 (6)</td><td>94.7 (2)</td><td>90.6 (7)</td></tr>
          <tr><td>VIRCHOW2</td><td>Histopathology</td><td>VM</td><td>82.3 (5)</td><td>88.0 (3)</td><td>89.1 (5)</td><td>92.6 (6)</td><td>88.0 (6)</td><td>87.2 (2)</td><td>90.8 (1)</td><td>92.0 (4)</td><td>93.9 (9)</td><td>91.0 (3)</td></tr>
          <tr><td>CONCH</td><td>Histopathology</td><td>VLM</td><td>75.1 (15)</td><td>84.5 (14)</td><td>81.7 (15)</td><td>91.1 (14)</td><td>83.1 (15)</td><td>82.1 (13)</td><td>87.9 (13)</td><td>87.3 (13)</td><td>91.0 (15)</td><td>87.1 (14)</td></tr>
          <tr><td>CONCH&nbsp;1.5</td><td>Histopathology</td><td>VLM</td><td>75.9 (14)</td><td>84.2 (15)</td><td>83.3 (12)</td><td>91.4 (10)</td><td>83.7 (14)</td><td>81.6 (14)</td><td>87.4 (15)</td><td>87.0 (15)</td><td>92.1 (14)</td><td>87.0 (15)</td></tr>
          <tr><td>KEEP</td><td>Histopathology</td><td>VLM</td><td>79.8 (9)</td><td>87.2 (8)</td><td>87.2 (9)</td><td>93.1 (4)</td><td>86.9 (9)</td><td>85.6 (11)</td><td>89.7 (9)</td><td>89.3 (10)</td><td>93.8 (10)</td><td>89.6 (10)</td></tr>
          <tr><td>MUSK</td><td>Histopathology</td><td>VLM</td><td>77.2 (12)</td><td>85.7 (11)</td><td>82.5 (14)</td><td>91.1 (13)</td><td>84.1 (13)</td><td>80.6 (15)</td><td>87.9 (14)</td><td>87.6 (12)</td><td>93.3 (12)</td><td>87.4 (13)</td></tr>
          <tr><td>PLIP</td><td>Histopathology</td><td>VLM</td><td>69.4 (17)</td><td>79.9 (16)</td><td>74.4 (16)</td><td>86.4 (16)</td><td>77.5 (16)</td><td>77.1 (18)</td><td>84.7 (19)</td><td>82.1 (17)</td><td>88.6 (16)</td><td>83.1 (17)</td></tr>
          <tr><td>QUILTNET</td><td>Histopathology</td><td>VLM</td><td>69.9 (16)</td><td>77.7 (19)</td><td>73.4 (17)</td><td>85.3 (17)</td><td>76.6 (17)</td><td>77.0 (19)</td><td>82.9 (21)</td><td>81.2 (20)</td><td>88.5 (18)</td><td>82.4 (19)</td></tr>
          <tr><td>DINOv2-B</td><td>Natural-image</td><td>VM</td><td>64.0 (20)</td><td>77.5 (20)</td><td>70.4 (20)</td><td>78.1 (20)</td><td>72.5 (20)</td><td>76.0 (20)</td><td>83.9 (20)</td><td>80.1 (21)</td><td>87.6 (21)</td><td>81.9 (21)</td></tr>
          <tr><td>DINOv2-L</td><td>Natural-image</td><td>VM</td><td>66.1 (19)</td><td>79.0 (17)</td><td>71.4 (19)</td><td>78.5 (19)</td><td>73.7 (19)</td><td>74.0 (21)</td><td>85.3 (16)</td><td>82.1 (16)</td><td>87.7 (20)</td><td>82.3 (20)</td></tr>
          <tr><td>ViT-B/16</td><td>Natural-image</td><td>VM</td><td>63.6 (21)</td><td>76.7 (21)</td><td>68.7 (21)</td><td>77.0 (22)</td><td>71.5 (21)</td><td>78.2 (17)</td><td>84.7 (18)</td><td>81.2 (19)</td><td>87.9 (19)</td><td>83.0 (18)</td></tr>
          <tr><td>ViT-L/16</td><td>Natural-image</td><td>VM</td><td>66.5 (18)</td><td>78.7 (18)</td><td>71.8 (18)</td><td>81.8 (18)</td><td>74.7 (18)</td><td>79.3 (16)</td><td>85.1 (17)</td><td>81.3 (18)</td><td>88.5 (17)</td><td>83.6 (16)</td></tr>
          <tr><td>CLIP-B/32</td><td>Natural-image</td><td>VLM</td><td>57.0 (23)</td><td>71.0 (23)</td><td>63.7 (23)</td><td>73.7 (23)</td><td>66.4 (23)</td><td>69.0 (23)</td><td>81.3 (23)</td><td>75.8 (23)</td><td>84.7 (23)</td><td>77.7 (23)</td></tr>
          <tr><td>CLIP-L/14</td><td>Natural-image</td><td>VLM</td><td>62.5 (22)</td><td>74.6 (22)</td><td>66.5 (22)</td><td>77.4 (21)</td><td>70.2 (22)</td><td>73.6 (22)</td><td>82.8 (22)</td><td>78.5 (22)</td><td>86.7 (22)</td><td>80.4 (22)</td></tr>
        <tbody>
    </table>
</div>

---

# 🏆 Zero-shot VLM Classification Leaderboard

F1-score on test sets of all supported datasets and average across datasets for the *zero-shot classification* task. Only VLM models with publicly released patch-level text encoders are included.

<div class="table-responsive-sm">
    <table id="zeroshotTable" class="table table-hover table-bordered table-sm nowrap pivot w-100" style="width:100%">
        <thead class="align-middle text-center">
          <tr>
            <th>Model</th>
            <th>Domain</th>
            <th>Type</th>
            <th>bach</th>
            <th>bracs</th>
            <th>break_his</th>
            <th>ccrcc</th>
            <th>crc</th>
            <th>esca</th>
            <th>mhist</th>
            <th>patch_camelyon</th>
            <th>tcga_crc_msi</th>
            <th>tcga_tils</th>
            <th>tcga_uniform</th>
            <th>wilds</th>
            <th>spider_breast</th>
            <th>spider_colorectal</th>
            <th>spider_skin</th>
            <th>spider_thorax</th>
            <th>Avg</th>
        </tr>
      </thead>
        <tbody>
          <tr><td>CONCH</td><td>Histopathology</td><td>VLM</td><td>56.1 (3)</td><td>37.9 (2)</td><td>53.6 (1)</td><td>56.9 (2)</td><td>51.8 (4)</td><td>40.1 (1)</td><td>60.8 (2)</td><td>57.8 (3)</td><td>21.6 (4)</td><td>47.4 (5)</td><td>37.9 (2)</td><td>83.2 (2)</td><td>30.7 (3)</td><td>31.4 (3)</td><td>35.1 (3)</td><td>43.0 (3)</td><td>46.6 (3)</td></tr>
          <tr><td>KEEP</td><td>Histopathology</td><td>VLM</td><td>63.4 (1)</td><td>34.2 (3)</td><td>45.0 (3)</td><td>69.1 (1)</td><td>80.6 (1)</td><td>33.3 (2)</td><td>41.3 (7)</td><td>71.4 (1)</td><td>15.5 (6)</td><td>55.5 (2)</td><td>44.9 (1)</td><td>89.4 (1)</td><td>37.7 (1)</td><td>44.4 (1)</td><td>60.7 (1)</td><td>51.8 (2)</td><td>52.4 (1)</td></tr>
          <tr><td>MUSK</td><td>Histopathology</td><td>VLM</td><td>62.5 (2)</td><td>38.6 (1)</td><td>52.6 (2)</td><td>50.7 (3)</td><td>57.9 (3)</td><td>25.9 (4)</td><td>63.8 (1)</td><td>53.5 (5)</td><td>22.7 (3)</td><td>50.1 (3)</td><td>32.4 (3)</td><td>71.2 (3)</td><td>36.3 (2)</td><td>36.2 (2)</td><td>48.5 (2)</td><td>55.2 (1)</td><td>47.4 (2)</td></tr>
          <tr><td>PLIP</td><td>Histopathology</td><td>VLM</td><td>42.7 (4)</td><td>25.9 (5)</td><td>25.6 (5)</td><td>38.2 (5)</td><td>61.4 (2)</td><td>31.5 (3)</td><td>53.9 (5)</td><td>46.1 (7)</td><td>16.3 (5)</td><td>64.1 (1)</td><td>10.3 (5)</td><td>51.0 (6)</td><td>14.6 (4)</td><td>28.3 (4)</td><td>23.5 (4)</td><td>22.6 (4)</td><td>34.7 (4)</td></tr>
          <tr><td>QUILTNET</td><td>Histopathology</td><td>VLM</td><td>30.3 (5)</td><td>29.1 (4)</td><td>37.5 (4)</td><td>24.2 (6)</td><td>44.2 (5)</td><td>14.2 (5)</td><td>57.1 (4)</td><td>65.8 (2)</td><td>50.4 (1)</td><td>47.6 (4)</td><td>12.3 (4)</td><td>44.3 (7)</td><td>13.9 (5)</td><td>25.0 (5)</td><td>18.5 (5)</td><td>19.7 (5)</td><td>33.4 (5)</td></tr>
          <tr><td>CLIP-B/32</td><td>Natural-image</td><td>VLM</td><td>13.2 (7)</td><td>7.5 (7)</td><td>18.7 (6)</td><td>21.8 (7)</td><td>24.4 (7)</td><td>9.8 (7)</td><td>42.2 (6)</td><td>48.1 (6)</td><td>13.9 (7)</td><td>21.6 (7)</td><td>2.0 (7)</td><td>56.8 (5)</td><td>3.9 (7)</td><td>6.1 (7)</td><td>4.3 (7)</td><td>5.5 (6)</td><td>18.7 (7)</td></tr>
          <tr><td>CLIP-L/14</td><td>Natural-image</td><td>VLM</td><td>27.1 (6)</td><td>19.2 (6)</td><td>5.8 (7)</td><td>40.6 (4)</td><td>41.1 (6)</td><td>10.4 (6)</td><td>58.0 (3)</td><td>55.6 (4)</td><td>49.4 (2)</td><td>25.4 (6)</td><td>7.4 (6)</td><td>70.2 (4)</td><td>7.3 (6)</td><td>16.0 (6)</td><td>6.6 (6)</td><td>5.4 (7)</td><td>27.8 (6)</td></tr>
        <tbody>
    </table>
</div>
