\documentclass[conference]{IEEEtran}
        \IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{array}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{An Explainable AI Agent-Based Framework for\\
    Real-Time Multi-Class Violence Detection\\
    in Surveillance Videos}

\author{
\IEEEauthorblockN{Mohammad Nayeemuddin}
\IEEEauthorblockA{\textit{Dept. of Data Science.} \\
\textit{PVP Siddhartha Institute of Technology}\\
        Vijayawada, India \\
        22501A4439@pvpsit.ac.in}
\and
\IEEEauthorblockN{Puli Prabhas Reddy}
\IEEEauthorblockA{\textit{Dept. of Data Science.} \\
\textit{PVP Siddhartha Institute of Technology}\\
        Vijayawada, India \\
        22501A4452@pvpsit.ac.in}
\and
\IEEEauthorblockN{Parasa Ramya}
\IEEEauthorblockA{\textit{Dept. of Data Science.} \\
\textit{PVP Siddhartha Institute of Technology}\\
        Vijayawada, India \\
        23505A4406@pvpsit.ac.in}
\and
\IEEEauthorblockN{Paruchuri Jayasri}
\IEEEauthorblockA{\textit{Dept. of Data Science.} \\
\textit{PVP Siddhartha Institute of Technology}\\
        Hyderabad, India \\
        paruchurijayasri@pvpsiddhartha.ac.in}
}

\maketitle

% ─────────────────────────────────────────────────────────────────────────────
        \begin{abstract}
% ─────────────────────────────────────────────────────────────────────────────
In surveillance systems, violence detection is a critical aspect.
enhancing citizen safety and providing quick response to emergencies. Most
current deep learning methods are however restricted to binary.
classification of violent and non-violent activities, which decreases.
their capability to detect particular kinds of anomalies and provide.
interpretable outputs to operators. False alarms are also likely to occur in such systems.alarms, poor contextual knowledge, and poor generalization across. environments. This paper suggests a framework of explainable AI agents. to detect the multi-class violence in real-time in surveillance videos. The framework combines three AI components running in parallel:
YOLOv8n person detector and frame annotation, a fine tuned. TimesFormer model spatiotemporal action recognition on 14 crime. categories, and LLaMA 3.1~8B (accessed via Groq API) to generate. clear security warnings, with brief content. Python backend Multi-threaded. supports live input of webcams, RTSP IP cameras and pre-recorded. video files, having a thread-safe source-switching mechanism to avoid. race conditions. Experiments show 98.8\% test accuracy on completely invisible video following one fine-tuning era, an average inference latency. of 18.5~ms per sequence (54 FPS-equivalent), and almost flawless per-class. remember all high-priority crime categories. The natural-language explanation component directly deals with the transparency and trust gap. that is prevalent in current surveillance AI.
\end{abstract}

\begin{IEEEkeywords}
        violence detection, surveillance, multi-class, anomaly.
        explainable AI, YOLOv8, TimesFormer, spatiotemporal features, detection.
large language model, real-time processing, multi-threading.
\end{IEEEkeywords}

% ─────────────────────────────────────────────────────────────────────────────
        \section{Introduction}
% ─────────────────────────────────────────────────────────────────────────────

The increased use of surveillance cameras in the open places,
        transportation hubs, and commercial areas has made automated monitoring
        a research problem that is gaining prominence. In most real-world
        installations, a number of human operators are nevertheless anticipated to monitor various. live feeds real-time and this is not feasible or scalable.
        The shortcomings of manual monitoring increase with the increase in the number of cameras. get more visible--significant events are easily overlooked, and reaction. suffer as a consequence times. \cite{duja2024}.

        The automated video has advanced greatly through deep learning.
        and a number of systems have been created to be particularly useful to understand.identifying violence on video surveillance. But one significant drawback is. of the majority of the existing work is that it conceptualizes the problem as a binary. classification task, which involves merely making a judgment of a scene being violent or not~\cite{khan2025}. In reality, what kind of incident it is is very important. A robbery, an act of vandalism, and an explosion all require extremely different strategies by the security teams. Crude binary classifications are not very actionable particularly in situations where decisions have to be made in a hurry.

        One more problem that is less discussed in the literature is.
        \textit{explainability}.Even in the situation where a model is accurate in identifying a. threat, the security operators have frequently been shunned off with no clarification of why. the alarm was sounded or what they ought to do. When AI systems act being black boxes, they are likely to be mistrusted and underused in operational settings \cite{wang2025}.

        The practical deployment challenges also exist and are to be tackled by the existing research. rarely addresses. Numerous of the proposed systems have only been studied as off-line experiments. and are not shown in real-time, multi-source, environments. The capability to alternate webcam feeds, network-linked IP cameras and The use of a single running system to upload video files has not been well explored.

        The paper will fill these gaps by suggesting a framework that will bring.
        integrate a few components into a single integrated real-time system. The
        main contributions of this work are:

        \begin{itemize}
        \item A fine-tuned TimesFormer video model, which classifies surveillance.
        videos into 14 different categories of crime and anomaly going well.
        in addition to binary labels of most previous work.

        \item Real-time person detection with YOLOv8n, which labels every frame          with bounding boxes and gives a person. count in the alert                 generation step.

        \item An alert generation model (written in natural language) based on           LLaMA 3.1 8B. that transforms raw classification results into              concise, operational. security messages, so that the system can be         comprehensible to. non-technical operators.

        \item A Flask back-end, supporting three video input, running on several         threads. types (webcam, CCTV/RTSP, uploaded files) with clean source
        continuous and switching WebSocket video streaming to a. browser-based dashboard.
        \end{itemize}

        The rest of this paper is organized as follows. Section~II reviews related
        work. Section~III describes the overall system architecture. Section~IV
        describes the multi-threaded design. Section~V covers the methodology for each AI component. Section~VI presents experimental results. Limitations and future directions are discussed in Section~VII. The paper ends in Section~VIII.
        % ─────────────────────────────────────────────────────────────────────────────
        \section{Related Work}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Deep Learning Approaches to Violence Detection}

        Khan \textit{et al.}~\cite{khan2025} developed a three-stage pipeline for violence detection in industrial surveillance. In their approach, a MobileNet-SSD model first filters out frames that
        do not contain people. A C3D network then extracts spatiotemporal features
        from 50-frame sequences, and a Softmax classifier produces a binary violence
        label with real-time alerting. Their results improve on earlier baselines, but performance drops noticeably when the model is tested on datasets it was
        not trained on, which points to limited generalization. The binary output also means the system cannot distinguish between different types of incidents, and no explainability mechanism is included.

        Shoaib \textit{et al.}~\cite{shoaib2023} focused on reducing computational cost in violence detection by introducing two keyframe selection methods, called DeepKeyFrm and AreaDiffKey, to avoid processing redundant frames. They evaluated two classifiers: EvoKeyNet, which combines a CNN with evolutionary feature selection, and KFCRNet, which fuses CNN features with an ensemble of LSTM, Bi-LSTM, and GRU units. Testing across five datasets produced strong accuracy and AUC scores, though the method requires careful threshold tuning and does not generalize easily to highly dynamic scenes. Like other methods in this space, it lacks any component that explains predictions to human users.

        \subsection{Survey Perspectives on Anomaly Detection}

        Duja \textit{et al.}~\cite{duja2024}reviewed a broad range of deep
        learning approaches for anomaly detection in surveillance video, including
        reconstruction-based, prediction-based, and hybrid methods. They point out
        several recurring problems: poor cross-dataset generalization, a lack of large and realistic anomaly datasets, high computational cost, and the absence of agreed-upon evaluation standards. A separate machine learning survey~\cite{survey2023} echoes these findings and adds that rare or subtle anomalies are especially difficult to detect, and that real-time deployment remains a challenge for most proposed systems.

        \subsection{Explainability in Video-Based AI}

        Wang and Liu~\cite{wang2025} proposed
        STAA (Spatio-Temporal Attention Attribution), an XAI technique designed for
        Transformer-based video models. STAA derives spatial and temporal importance scores directly from the model's self-attention weights in a single
        forward pass, achieving a latency of around 150~ms. The method produces
        visual explanations in the form of attention heatmaps overlaid on video
        frames. While useful, these visualizations are targeted at model developers
        or researchers rather than security operators who need to act quickly. In
        operational settings, a natural-language message that states the threat and
        recommends an action is generally more practical than an attention map.

        \subsection{Summary of Gaps}

        Looking across these works, three gaps stand out clearly. First, almost all
        systems produce only binary outputs, which limits how useful they are in real deployments where the type of incident determines the response.
        Second, natural-language explainability aimed at non-technical end users is essentially absent from the literature. Third, no existing work demonstrates a live, multi-source system that handles webcam, IP camera, and file-based input within a single, production-ready application. The framework proposed in this paper is designed to address all three of these shortcomings.
        % ─────────────────────────────────────────────────────────────────────────────
        \section{System Architecture}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Overview}

        The proposed system is a full-stack, multi-threaded web application built on a Python Flask backend. WebSocket support is provided via the Flask-Sock library, which allows the server to push annotated video frames continuously to the browser without the overhead of repeated HTTP requests. Three AI models run concurrently in separate threads and share a common frame buffer. Their combined outputs are merged into a single result dictionary that the frontend polls via a REST endpoint, while the video stream is delivered over a persistent
        WebSocket connection.
        Fig.~\ref{fig:architecture} illustrates the overall data flow.

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{fig_architecture.png}
        \caption{Overall system architecture. Video from one of three source
        adapters is written into a shared frame buffer. YOLOv8n, TimesFormer,
        and LLaMA run concurrently as background threads. The annotated video
        stream is pushed to the browser over WebSocket, while classification
        results are retrieved through a REST polling endpoint.}
        \label{fig:architecture}
        \end{figure}

        \subsection{Video Input Sources}

        The system consumes video streams from three different kinds of sources using separate adapters for each one.

        \textbf{Webcam.} The local camera is opened using OpenCV's
        \texttt{VideoCapture(0)}. The internal frame buffer size is set to~1
        to ensure the system always processes the most recent frame rather than
        accumulating stale frames. The capture runs at 640$\times$480 pixels at
        up to 30~FPS.

        \textbf{CCTV / IP Camera (RTSP).} Rather than relying on OpenCV's
        built-in RTSP reader, which tends to fail with non-standard camera
        firmware or when passwords contain special characters, the system uses
        FFmpeg as an external subprocess. FFmpeg is launched with TCP transport
        (\texttt{-rtsp\_transport tcp}), which is more reliable than the default
        UDP on typical network setups. The decoded frames are written as raw
        BGR24 pixels to FFmpeg's standard output pipe, and a dedicated Python
        thread reads exactly $640 \times 480 \times 3 = 921{,}600$~bytes per
        frame, converts the byte buffer to a NumPy array, and reshapes it into
        a standard OpenCV-compatible frame. Any camera credentials are masked
        in log output before they are written.

        \textbf{Uploaded Video.} An uploaded MP4 (or similar) file is saved to
        disk and decoded by a background thread using \texttt{cv2.VideoCapture}.
        All frames are loaded into memory for loop playback. For longer videos
        (over 3000 frames), every other frame is skipped to keep memory usage
        manageable while maintaining the correct playback speed.

        \subsection{Frontend Dashboard}

        The browser displays the live annotated video feed along with the current label for activity being performed, confidence value, number of people in the view, severity level, LLM-generated alert text, and event log timestamp. Severity is shown using a color-coding overlay composite on the video frame in the following way: the red banner indicates high-danger events, orange banner indicates a warning level event, and green banner labels normal
        scenes. Fig.~\ref{fig:dashboard} shows the dashboard layout.

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{fig_dashboard.png}
        \caption{The browser dashboard features the annotated live video pane (left), real-time detection result along with a confidence meter and LLM-generated alert text (right), and event log timeline (bottom).}
        \label{fig:dashboard}
        \end{figure}

        % ─────────────────────────────────────────────────────────────────────────────
        \section{Multi-Threaded Pipeline}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Thread Design}

        The backend runs five concurrent threads. Each thread has a single,
        well-defined responsibility, and all shared state is protected through
        eight named \texttt{threading Lock} objects. A \texttt{threading Event}
        object called \texttt{stop\_event}  acts as a globally accessible signal that a thread should check before starting its next loop iteration.
        Table~\ref{tab:threads} shows a brief description of each thread along with its timing.

        \begin{table}[htbp]
        \caption{Concurrent Thread Responsibilities and Timing}
        \label{tab:threads}
        \centering
        \renewcommand{\arraystretch}{1.25}
        \begin{tabular}{|p{2.0cm}|p{1.3cm}|p{4.1cm}|}
        \hline
        \textbf{Thread} & \textbf{Interval} & \textbf{Responsibility} \\
        \hline
        \texttt{capture\_thread} & Continuous &
        Reads webcam frames via OpenCV; stores the latest frame in
        \texttt{raw\_frame} and appends to \texttt{pred\_buf} (capped at
        96 frames). \\
        \hline
        \texttt{ffmpeg\_capture} & Continuous &
        Reads raw BGR bytes from FFmpeg's stdout for RTSP sources and
        reconstructs them into OpenCV-compatible NumPy frames. \\
        \hline
        \texttt{yolo\_thread} & 250~ms &
        Runs YOLOv8n on the current frame; draws person bounding boxes and
        overlays the latest activity label with severity color coding;
        writes the result to \texttt{ann\_frame}. \\
        \hline
        \texttt{predict\_thread} & 5~s &
        Samples 16 frames from \texttt{pred\_buf}; runs TimesFormer
        inference; updates the shared result dictionary; triggers
        LLM alert generation when the predicted label changes. \\
        \hline
        \texttt{video\_ws} & 25~FPS target &
        JPEG-encodes and Base64-transmits the annotated frame to the
        browser over WebSocket, paced to hit the target frame rate. \\
        \hline
        \end{tabular}
        \end{table}

        \subsection{Source Switching and Thread Safety}

        On user switching the current video source or pressing the stop button, a function \texttt{\_hard\_stop()} is executed while holding a master lock object.
        The process is as follows:

        \begin{enumerate}
        \item \texttt{stop\_event.set()} stops all threads  work at the beginning               of the next loop iteration.

        \item A \texttt{source\_id}  incrementing the atomic source id counter,                   any WebSocket sender thread, created before for a different                source, still keeps a copy of it in memory and quits the                   loop when noticing a discrepancy.

        \item release the OpenCV capture object, stop and terminate any FFmpeg           process associated with the CCTV camera;

        \item release all shared frame buffers and prediction buffers;.

        \item waiting for another 100 ms is enough for WebSocket threads to
        observe the updated source ID---\texttt{stop\_event.clear()}
        re-arms the processing threads for the new source.
        \end{enumerate}

        All threads have been started with the parameter \texttt{daemon=True}. This
        meaning that they will be automatically stopped after termination of the Flask server process, which makes it possible to terminate all processes without finishing any infinite loops.

        % ─────────────────────────────────────────────────────────────────────────────
        \section{Methodology}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Person Detection Using YOLOv8n}

        Person detection is handled by YOLOv8 Nano~\cite{jocher2023}, It is used because of its favorable trade-off between precision and latency. This model runs with its default COCO pre-trained weights without any further fine-tuning since detection of people in surveillance video clips is well within the scope of its training distribution. The inference is run at a low resolution of 320×320 pixels to achieve 250 ms latency along with the parallel TimesFormer inference stream. Only detections of objects belonging to COCO class 0 (person) are kept, and green boxes and number Np of persons are drawn on the video frame. Np is also provided in the prompt for the generated alert so that it contains information about the number of detected individuals. Severity-colored visual layers are created:

        \begin{equation}
        \text{overlay}(f) =
        \begin{cases}
        \text{red banner} & \text{if severity} = \textsc{Danger} \\
        \text{orange banner} & \text{if severity} = \textsc{Warn} \\
        \text{green text} & \text{if severity} = \textsc{Normal}
        \end{cases}
        \end{equation}

        \subsection{Multi-Class Action Recognition Using TimesFormer}

        For temporal action understanding, the framework uses
        TimesFormer~\cite{bertasius2021}, A Transformer-based architecture is used, performing different attention mechanisms on spatial and temporal slices. Such design allows modeling fine-grained appearance features along with temporal dynamics in the sequence, resulting in better person detection and tracking performance.

        The backbone is the publicly available pre-trained checkpoint
        \texttt{facebook/timesformer-base-finetuned-k400},This particular Transformer model was trained on Kinetics-400 dataset. Its original head performing classification into 400 classes is swapped for a simple linear layer mapping 768-dimensional CLS token embedding into 14 class probabilities:

        \begin{equation}
        \hat{\mathbf{y}} = \mathbf{W}\,\mathbf{h}_\text{CLS} + \mathbf{b},
        \quad \mathbf{W} \in \mathbb{R}^{14 \times 768}.
        \end{equation}

        The model is fine-tuned on a dataset covering the 14 categories listed
        in Table~\ref{tab:labels}. Weight loading uses \texttt{strict=False}
        to accommodate the replaced head without errors. During inference,
        16 frames are sampled at uniform temporal stride from the available buffer:

        \begin{equation}
        \text{step} = \lfloor |\mathcal{B}| / 16 \rfloor, \quad
        \mathcal{S} = \{ f_{\,k \cdot \text{step}} \}_{k=0}^{15},
        \end{equation}

        where $|\mathcal{B}|$ is the current buffer size. Each sampled frame is
        converted from BGR to RGB, resized to $224\times224$, and normalized into
        a PyTorch tensor, producing an input of shape $[1,\;16,\;3,\;224,\;224]$.
        The predicted class and confidence score are:

        \begin{equation}
        \hat{c} = \arg\max_j\; p_j, \quad
        \hat{p} = \max_j\; \text{softmax}(\hat{\mathbf{y}})_j.
        \end{equation}

        Inference is wrapped in \texttt{torch.no\_grad()} to suppress gradient
        computation, which roughly doubles throughput and reduces memory usage
        compared to running in training mode.

        \begin{table}[htbp]
        \caption{14-Class Label Set and Severity Tiers}
        \label{tab:labels}
        \centering
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{|l|l|}
        \hline
        \textbf{Severity Tier} & \textbf{Categories} \\
        \hline
        \textsc{Danger} & Abuse, Assault, Explosion, Fighting, \\
        & Shooting, Arson, Robbery \\
        \hline
        \textsc{Warn}   & Arrest, Burglary, RoadAccidents, \\
        & Shoplifting, Stealing, Vandalism \\
        \hline
        \textsc{Normal} & NonViolence \\
        \hline
        \end{tabular}
        \end{table}

        \subsection{Natural-Language Alert Generation Using an LLM}

        In order to make the model accessible for non-expert users within the security team, the proposed architecture includes the creation of alerts using LLaMA 3.1~8B~\cite{meta2024} through the Groq inference API. Instead of only providing the activity label and its probability score, the system will return a natural language message stating the detected activity and the recommended response.

        The LLM will be called only if there is a change in the predicted activity label during inference. The prompt will take four inputs, which include the activity type $\hat{c}$, number of people $N_p$, probability score $\hat{p}$, and a label indicating whether the threat is critical or warning. There is an instruction given by the system that limits the response to two sentences and forces the response to start with the label provided before requesting the recommended action. The generation temperature is set to 0.2 to ensure consistency. If the predicted activity label is NonViolence, then the message generated will be a static message and not call the API..

        % ─────────────────────────────────────────────────────────────────────────────
        \section{Experimental Results}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Classification Performance}

        The optimized TimesFormer model was evaluated on an unseen test dataset that did not contain any video data from the train and validation datasets. Confusion matrix results for all 14 classes are provided in Fig. \ref{fig:confusion}.

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{fig2_confusion_matrix.png}
        \caption{Confusion matrices for the 14-class TimesFormer model (raw counts
        on the left, row-normalized on the right). The model is able to perfectly recall 12 out of 14 classes. The few cases of classification errors happen when there are confusions between Abuse and RoadAccidents (5\%), and NonViolence and Fighting (5\%). Notably, no Danger class sample is wrongly classified as NonViolence.}
        \label{fig:confusion}
        \end{figure}

        The model achieves a recall score of 1.00 for 12 out of 14 categories, covering all Danger category classes. Two categories which achieve recall scores that are slightly less than perfect scores include Abuse (0.95) and NonViolence (0.95). The reason behind this is that their motion patterns can easily be confused with each other. However, it is ensured that the primary safety invariant holds true.

        \subsection{Generalization}

        Fig.~\ref{fig:gengap} compares training accuracy, validation accuracy,
        and true test accuracy. The test set is a video-level holdout with no
        overlap with any training data.

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{fig4_generalization_gap.png}
        \caption{Training accuracy (97.7\%), validation accuracy (98.6\%), and
        test accuracy on completely unseen videos (98.8\%, hatched bar). The
        negligible gap between validation and test accuracy indicates that the
        model generalizes well without overfitting.}
        \label{fig:gengap}
        \end{figure}

        Upon completion of one single epoch of fine-tuning, the model reaches accuracy scores of 97.7\%, 98.6\%, and 98.8\% respectively on the train, validation, and test sets comprising fully unseen samples. The slight improvement of the test accuracy compared to the validation accuracy suggests that pretraining on Kinetics-400 offers a solid foundation of features that can be transferred successfully to this problem without overfitting to the fine-tuning dataset.

        \subsection{Inference Latency}

        Fig.~\ref{fig:latency} shows the per-sequence latency distribution and
        a stability trace measured over 270 consecutive inferences with the
        warmup period excluded.

        \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{fig5_inference_latency_fixed.png}
        \caption{TimesFormer inference latency distribution (left) and steady-state
        stability trace over 270 sequences (right). Mean latency is 18.5~ms,
        P95 is 19.9~ms, and the throughput-equivalent is 54~FPS. Occasional
        spikes stay below 25~ms.}
        \label{fig:latency}
        \end{figure}

        On average, the model inference time per batch size 16 equals 18.5 ms with P95 equal to 19.9 ms which implies a throughput of 54 frames per second. The latency graph shows steady performance during all 270 inference cycles, with occasional spikes that stay under 25 ms. As for predict thread that runs in 5-second intervals, the video streaming thread will remain unaffected on any device.

        \subsection{Performance Summary}

        Table~\ref{tab:summary} brings together the key metrics from these
        experiments.

        \begin{table}[htbp]
        \caption{Performance Metrics Summary}
        \label{tab:summary}
        \centering
        \renewcommand{\arraystretch}{1.2}
        \begin{tabular}{|l|c|l|}
        \hline
        \textbf{Metric} & \textbf{Value} & \textbf{Notes} \\
        \hline
        Train Accuracy (Epoch 1) & 97.7\%          & Single epoch fine-tuning \\
        Validation Accuracy      & 98.6\%          & Seen video split \\
        Test Accuracy (Unseen)   & \textbf{98.8\%} & No train/test overlap \\
        Mean Inference Latency   & 18.5~ms         & Per 16-frame sequence \\
        P95 Latency              & 19.9~ms         & Warmup excluded \\
        FPS-Equivalent           & 54.0            & Based on mean latency \\
        Perfect-Recall Classes   & 12~/~14         & Includes all 7 Danger classes \\
        Classification Head      & 14-class        & Linear layer on CLS token \\
        \hline
        \end{tabular}
        \end{table}

        % ─────────────────────────────────────────────────────────────────────────────
        \section{Discussion}
        % ─────────────────────────────────────────────────────────────────────────────

        \subsection{Strengths}

        Among the promising results is the impressive generalization capability of the model after a single epoch of fine-tuning. Reaching 98.8\% accuracy on completely unknown videos shows that the spatiotemporal features learned by the model on Kinetics-400 can be transferred to videos from the surveillance camera domain, meaning that vast amounts of domain-specific training data are not necessary for satisfactory results. Mean latency of 18.5 ms ensures real-time capabilities and allows for scaling to many cameras simultaneously.

        The most practically valuable observation is that there are no misclassifications of Danger class events into NonViolence in any of the tested video sequences. This is a crucial requirement for systems designed for potential use in real-life security applications because failing to detect a danger usually causes higher expenses than an unnecessary alarm. For binary systems, it is impossible to ensure this safety guarantee.

        The natural-language alert component fills a gap that purely
        classification-based systems cannot address. Telling an operator
        ``\textsc{[DANGER]} Fighting detected involving 3 persons. Contact
        security personnel and initiate evacuation protocol immediately'' is
        more useful than simply displaying a label. This is especially true in
        high-pressure situations where operators need to act without pausing to
        interpret raw model outputs.

        \subsection{Limitations}

        There are also some limitations that must be noted. First, the fine tuning was done using only one epoch on a relatively small dataset. The two classes that did not show ideal recall rates, Abuse, and NonViolence, might need more training data, more epochs, and perhaps even some data augmentation. The use of a large language model (LLM) adds an external dependency and also increases latency time (around 200–500 ms per alert) which is not included in the numbers reported by TimesFormer. Switching from a cloud API to a quantized version hosted locally would allow making the application independent of any outside elements and removing this source of delay. The RTSP URL used at the moment follows the standard for Dahua cameras and would need to be modified for other brands.

        \subsection{Future Directions}

        There are several ways that could be pursued. For example, by increasing the training period for more epochs with data augmentation and balanced samples per class, it will resolve the issue of two imperfect recall classes, as well as increase robustness in difficult situations. By replacing the Groq API call with a local installation of the quantized large language model, such as a 4-bit GGUF model, we can decrease dependence on an external library and reduce the alert generation time considerably. Adding the overlay of spatial attention with the text alert will give operators the ability to locate the detected behavior in the frame. Scaling out the application to support simultaneous processing of multiple camera feeds through the same scheduling queue is another straightforward step. Running a user study involving security professionals will help gauge the benefit of using natural language alerts compared to purely label-based alerts.

        % ─────────────────────────────────────────────────────────────────────────────
        \section{Conclusion}
        % ─────────────────────────────────────────────────────────────────────────────

        This paper introduces an explainable AI agent framework for detecting violence in surveillance video feeds in real time using multiple classifications. The framework uses YOLOv8n to detect people, fine-tuned TimesFormer to detect 14 types of actions, and LLaMA 3.1 8B to generate natural-language alerts for the identified events. Overall, these three elements solve the problems posed by the binary-classification problem, the lack of explainability, and multi-source integration that appear repeatedly in surveillance AI literature.

        The performance is impressive as well. It is able to recognize events on previously unseen videos with 98.8\% accuracy after just one epoch of fine-tuning with a latency of 18.5 ms and perfect recall on all seven high-danger categories. It also retains decoupling between video feed, AI processing, and human interaction, meaning none of those interfere with each other.

        However, the most notable advantage is not any single result but the combination thereof. The framework allows to know what is happening, what entities are involved, and what measures must be taken in regard to a situation captured by the camera in real time. This research seeks to create a foundation for creating a surveillance system with practical application.

        % ─────────────────────────────────────────────────────────────────────────────
        \section*{Acknowledgment}
        % ─────────────────────────────────────────────────────────────────────────────

        The authors acknowledge the faculty of the Department of Computer Science
        and Engineering for their constructive feedback during the development
        and review phases of this work.

        % ─────────────────────────────────────────────────────────────────────────────
        \begin{thebibliography}{10}
        % ─────────────────────────────────────────────────────────────────────────────

        \bibitem{khan2025}
        H.~Khan, X.~Yuan, L.~Qingge, and K.~Roy,
        ``Violence detection from industrial surveillance videos using deep
        learning,''
        \textit{IEEE Transactions on Industrial Informatics}, 2025.
        doi:10.1109/10844266.

        \bibitem{wang2025}
        Z.~Wang and Y.~Liu,
        ``STAA: Spatio-temporal attention attribution for real-time interpreting
        transformer-based AI video models,''
        \textit{IEEE Transactions on Pattern Analysis and Machine Intelligence},
        2025.
        doi:10.1109/11020668.

        \bibitem{duja2024}
        K.~U.~Duja, I.~A.~Khan, and M.~Alsuhaibani,
        ``Video surveillance anomaly detection: A review on deep learning
        benchmarks,''
        \textit{IEEE Access}, 2024.
        doi:10.1109/10744017.

        \bibitem{shoaib2023}
        M.~Shoaib, A.~Ullah, I.~A.~Abbasi, F.~Algarni, and A.~S.~Khan,
        ``Augmenting the robustness and efficiency of violence detection systems
        for surveillance and non-surveillance scenarios,''
        \textit{IEEE Access}, 2023.
        doi:10.1109/10304142.

        \bibitem{survey2023}
        ``A comprehensive survey of machine learning methods for surveillance
        videos anomaly detection,''
        \textit{IEEE Access}, 2023.
        doi:10.1109/10271300.

        \bibitem{bertasius2021}
        G.~Bertasius, H.~Wang, and L.~Torresani,
        ``Is space-time attention all you need for video understanding?''
        in \textit{Proc. Int. Conf. Machine Learning (ICML)}, 2021.

        \bibitem{jocher2023}
        G.~Jocher \textit{et al.},
        ``Ultralytics YOLOv8,'' 2023.
        [Online]. Available: \texttt{https://github.com/ultralytics/ultralytics}

        \bibitem{meta2024}
        Meta AI,
        ``LLaMA 3.1: Open foundation and fine-tuned chat models,'' 2024.
        [Online]. Available: \texttt{https://ai.meta.com/llama/}

        \end{thebibliography}

        \end{document}