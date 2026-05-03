# 参考文献清单

说明：以下参考文献用于支撑“智慧课堂行为检测、多模态音视频理解、ASR 质量门控、跨模态对齐、可视分析系统”五条论证线。正式投稿前建议按目标模板转换为 BibTeX 或 GB/T 7714 格式。

## 核心必引

1. Radford, A., Kim, J. W., Hallacy, C., et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML, 2021.  
   链接：https://arxiv.org/abs/2103.00020  
   用途：支撑视觉-文本语义对齐思想。

2. Radford, A., Kim, J. W., Xu, T., et al. **Robust Speech Recognition via Large-Scale Weak Supervision**. arXiv, 2022.  
   链接：https://arxiv.org/abs/2212.04356  
   用途：支撑本地 Whisper ASR 与质量门控。

3. Li, G., Wei, Y., Tian, Y., et al. **Learning To Answer Questions in Dynamic Audio-Visual Scenarios**. CVPR, 2022.  
   链接：https://openaccess.thecvf.com/content/CVPR2022/html/Li_Learning_To_Answer_Questions_in_Dynamic_Audio-Visual_Scenarios_CVPR_2022_paper.html  
   用途：支撑音视频场景理解与动态事件语义。

4. Zhai, Y., Wang, L., Tang, W., et al. **MAViL: Masked Audio-Video Learners**. NeurIPS, 2023.  
   链接：https://papers.nips.cc/paper_files/paper/2023/hash/4f6fa56d6f0e5f4874f2ec5cb903caeb-Abstract-Conference.html  
   用途：支撑音视频自监督学习与多模态表征。

5. Joannou, A., et al. **Audiovisual Moments in Time: A Large-Scale Audiovisual Event Dataset**. PLOS ONE, 2024.  
   链接：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0301098  
   用途：支撑音视频事件检测数据集与多模态事件建模。

6. Li, et al. **Multimodal Audio-Visual Detection in Classroom**. Scientific Reports, 2025.  
   链接：https://www.nature.com/articles/s41598-025-00588-0  
   用途：直接支撑课堂音视频事件检测背景。

## 课堂行为检测与 YOLO 系列

7. **Classroom Behavior Detection Based on Improved YOLOv5 Algorithm Combining Multi-Scale Feature Fusion and Attention Mechanism**. Applied Sciences, 2022.  
   DOI：https://doi.org/10.3390/app12136790  
   用途：课堂行为检测视觉-only 基线。

8. **MSTA-SlowFast: A Student Behavior Detector for Classroom Environments**. Sensors, 2023.  
   DOI：https://doi.org/10.3390/s23115205  
   用途：课堂行为时空建模基线。

9. **Student Behavior Detection in the Classroom Based on Improved YOLOv8**. Sensors, 2023.  
   DOI：https://doi.org/10.3390/s23208385  
   用途：YOLO 在课堂密集行为检测中的相关工作。

10. Wang, C.-Y., Bochkovskiy, A., Liao, H.-Y. M. **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**. CVPR, 2023.  
    链接：https://openaccess.thecvf.com/content/CVPR2023/html/Wang_YOLOv7_Trainable_Bag-of-Freebies_Sets_New_State-of-the-Art_for_Real-Time_Object_Detectors_CVPR_2023_paper.html  
    用途：YOLO 系列实时检测背景。

11. Wang, A., Chen, H., Liu, L., et al. **YOLOv10: Real-Time End-to-End Object Detection**. arXiv, 2024.  
    链接：https://arxiv.org/abs/2405.14458  
    用途：实时检测器发展趋势，对比 YOLO11 选型。

## 视频理解与跨模态扩展

12. Bertasius, G., Wang, H., Torresani, L. **Is Space-Time Attention All You Need for Video Understanding?** ICML, 2021.  
    链接：https://arxiv.org/abs/2102.05095  
    用途：Transformer 视频时空建模基线。

13. Zhang, C.-L., Wu, J., Li, Y. **ActionFormer: Localizing Moments of Actions with Transformers**. ECCV, 2022.  
    链接：https://arxiv.org/abs/2202.07925  
    用途：动作时序定位与事件片段建模。

14. Zhang, H., Li, X., Bing, L. **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding**. arXiv, 2023.  
    链接：https://arxiv.org/abs/2306.02858  
    用途：未来工作中可讨论的大模型音视频理解方向。

15. Wang, Y., et al. **InternVideo2: Scaling Foundation Models for Multimodal Video Understanding**. arXiv, 2024.  
    链接：https://arxiv.org/abs/2403.15377  
    用途：多模态视频基础模型上限参考。

16. Ye, et al. **CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios**. arXiv, 2024.  
    链接：https://arxiv.org/abs/2408.02282  
    用途：音视频问答与动态事件理解。

17. Chowdhury, et al. **Meerkat: Audio-Visual Large Language Model for Grounding in Space and Time**. arXiv, 2024.  
    链接：https://arxiv.org/abs/2410.00846  
    用途：音视频 grounding 与时空定位讨论。

## 本文引用策略

主文建议控制在 18 到 25 篇参考文献。优先引用 1-11；12-17 放在相关工作和未来展望。不要把未实际使用的 LLM 或 foundation model 写成本文方法，只能写成“相关工作/未来扩展”。

