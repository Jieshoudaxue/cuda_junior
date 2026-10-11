@[TOC](CUDA C++ 高效入门第一章 -- 为什么要学习 GPU + CUDA )
# 1 资料
本文是 CUDA C++ 高效入门系列第一篇，主要介绍 GPU + CUDA 的背景知识，我们将梳理 CUDA 技术在计算机技术世界中的位置，为大家学习 CUDA 提供一个全局视角。
在这个过程中，我们将回答：传统 CPU 程序员为什么要学习 GPU + CUDA，或者更具体一点，回答：传统嵌入式软件开发者，为什么要学习 GPU + CUDA C++。
本文参考资料如下：
（1）[GPU Survival Toolkit for the AI age: The bare minimum every developer must know](https://journal.hexmos.com/gpu-survival-toolkit/)
（2）[If GPUS Are So Good, Why Do We Still Use CPUs At All?](https://codingstuff.substack.com/p/if-gpus-are-so-good-why-do-we-still)
（3）[北京理工大学课程：并行编程原理与实践](https://www.xuetangx.com/course/bit0809bt0186/29601862)
（4）[cuda-programming-guide](https://bearneck.github.io/cuda-programming-guide-zh/chapters/01-introduction/) 1.1 节
本系列博客汇总链接：[CUDA C++ 高效入门系列](https://blog.csdn.net/cy1641395022/article/details/162685283)。
# 2 正文
## 2.1 什么是并行编程，为什么要并行编程
（1）并行编程和串行编程：并行编程的对立概念是串行编程。串行编程是一个 CPU 处理核一次处理一个任务，处理完再处理下一个任务。并行编程就是多个 CPU 处理核同时处理多个任务。
（2）生活类比：生活中串行和并行的情况都有。以盖一栋楼为例，必须先打地基，然后是地面建筑，最后是室内装潢，后一步依赖前一步，整体表现为串行执行。但在具体的某一个环节中，可以安排很多工种同时工作，该搬砖的搬砖，该砌墙的砌墙，并行完成一个大任务。
（3）为什么要并行编程：简单来说是为了效率。现实生活中，能并行的环节如果串行执行，必然效率低下，令人无法接受。典型的例子就是让一个人独立完成建一栋楼的所有工作，那真是又累又慢心里又苦。而如果是一个工程队来搞，说说笑笑间一栋楼就起来了。计算机作为现实世界的模拟，也是一样的道理，很多任务可以并行执行，提升效率，那么技术就会实现并行执行，让软件提升效率。
（4）计算机硬件：计算机系统的发展趋势，就是不断增加并行编程的效率。
单核变多核：在 2005 年以后，因为功耗和散热的问题，CPU 就从单核改为了多核（如下图），典型的产品是英特尔从奔腾系列改为了酷睿系列。
同构变异构：尽管有了多核 CPU，但是很多计算密集的任务（图像处理）很难使用 CPU 进一步提升效率，于是就有了 GPU（Graphics Processing Unit，图形处理器）。同时存在 CPU + GPU 的计算机就是异构计算机系统。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb5aca312cee4f3b83c6eecc5de74016.png =600x250)


## 2.2 CPU 和 GPU 101
（1）CPU：Central Processing Unit，中央处理器。它的设计目标就是为了解决一系列串行操作，即顺序执行任务。虽然 2005 年以后，人们努力通过多核提升并行效率，但 CPU 的基本设计理念还是优先考虑顺序执行。
（2）GPU：Graphics Processing Unit，图形处理器。GPU 专为高度并行计算而设计，将更多晶体管用于数据处理单元（下图绿色），而 CPU 为了完成复杂的任务，则将更多晶体管用于片上缓存和流控制。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ce7f9f4ca62749908b826ea88e5b4d79.png =600x270)
（3）CPU 与 GPU 的关系：相比 CPU ，GPU 强化了处理核，有极强的并行处理能力，但 GPU 弱化了流控制，因此他必须与 CPU 合作才能完成一件任务，而不是彻底取代 CPU。通常情况下，程序由 CPU 启动，CPU 担当调度师完成串行任务，GPU 分担其中大规模并行计算任务，比如图形计算，AI 模型推理等。具体参考：[If GPUS Are So Good, Why Do We Still Use CPUs At All?](https://codingstuff.substack.com/p/if-gpus-are-so-good-why-do-we-still)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a585ddbe603b40f18deb377026976415.png =600x300)
（4）从 GPU 到 GPGPU：
第一，最初的 GPU 是专门加速实时3D渲染中并行操作的固定功能硬件，简称显卡，不支持可编程。
第二，经过几代发展，到 2003 年，GPU 的可编程性就已经很强大了。
第三，到 2006 年，英伟达推出 CUDA （Compute Unified Device Architecture，统一计算设备架构）语言以后，GPU 计算已被用于加速几乎所有类型的计算工作负载，从流体动力学或能量传输等科学模拟，到数据库和分析等商业应用。这会的 GPU 也被成为 GPGPU（General-Purpose computing on Graphics Processing Units，图形处理器通用计算）。
第四，得益于 GPU 带来的算力提升，2012年，由Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton组成的团队提交的 AlexNet 深度卷积神经网络（CNN）模型，在著名的 ImageNet 图像识别挑战赛（ILSVRC）中一战成名，引爆了后来的 AI 革命，而 AI 革命也引爆了英伟达的市值。
（5）GPU 市场变化：早期显卡 GPU 市场基本是英伟达，AMD 和英特尔三分天下，但目前英特尔是 GPGPU 绝对的霸主，市占率约 90%，连带着 CUDA 也变成 GPGPU 编程的事实标准了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/70736ffc6380424b8dccf09b6a423f97.png =600x400)
（6）ASIC ，FPGA 和 GPGPU：在计算设备中，还有两个技术路线，一是 ASIC（专用集成电路），二是 FPGA（现场可编程门阵列）。前者是不可编程的固化集成电路，执行固定的计算任务，能效最高，但最不灵活，比如通信行业的交换机芯片。FPGA 支持使用 Verilog 进行编程，能效介于 ASIC 和 GPGPU 之间，但编程易用性远不如 GPGPU，上手门槛还是太高。最终，GPGPU 凭借其高并行吞吐量，以及 CUDA 等成熟软件生态带来的优秀可编程性，在 AI 加速计算领域取得了主导地位。
## 2.3 CPU 并行编程技术梳理
（1）多线程编程：对于 Linux C 开发者，主要使用操作系统提供的 pthread 系列接口。对于 C++ 开发者，主要使用 std::thread、std::mute 这些标准库，细节请参考：[cppreference](https://zh.cppreference.com)。
（2）OpenMP 编程：这是一种编译指导语句式的并行编程模型，底层基于多线程技术实现，常用于让循环并行化。OpenMP 具有操作简单，可移植性强的优点，缺点是依赖编译器实现，不灵活，专业程序员用的也不多（我几乎不用）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6e4efedca1b64a70abfcc706f032dd2e.png =600x300)![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c2a5ca09c2f447794da79e11c1bfa78.png =600x300)
（3）MPI 编程：MPI 即 Message Passing interface，也叫消息传递接口。前面的多线程编程和 OpenMP 编程都是单进程下的并行编程技术，整个程序只有一个内存地址空间，而 MPI 是一种多进程编程技术，每个进程有独立的内存地址空间。多进程编程通常使用 socket 接口，但比较麻烦，MPI 在此基础上进行了很好的封装，为程序员提供了一个更上层的接口，细节参考：[北京理工大学课程：并行编程原理与实践之 MPI 编程](https://www.xuetangx.com/course/bit0809bt0186/29601862)。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df68fd6acfe24d618f5a9a4622b7ac2c.png =600x300)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/696e841b92164ca5a7debd9069ed3093.png =600x300)
## 2.4 CPU + GPU 异构编程技术梳理
（1）CUDA：Compute Unified Device Architecture，统一计算设备架构，是英伟达针对自家 GPU 推出的异构编程技术。得益于英伟达在 GPU 市场的垄断地位，CUDA 已经成为 GPU 加速运算的首选编程语言/框架。CUDA 支持 C，C++，Fortran，Python，因此也分别叫做 CUDA C，CUDA C++，CUDA Fortran，CUDA Python。如果是嵌入式 GPU 编程，搞智驾机器人，CUDA C++ 的使用率远远大于 CUDA C，是绝对的首选。如果是桌面 GPU 编程，搞 AI 应用，CUDA Python 是绝对的首选，这主要得益于 Pytorch 的广泛使用。至于 CUDA Fortran 主要用于超级计算机和数据中心级别的科学计算，目标用户是拥有特定领域深厚知识（如流体力学、量子化学）的科学家。
（2）OpenCL：Open Computing Language，最早由苹果提出，各大巨头都参与后，由非营利组织 Khronos Group 维护，这是一种面向各种异构系统的统一编程框架（CPU，GPU，DSP，FPGA，ASIC）。但是种种原因，OpenCL 最后败给了英伟达自家的 CUDA，具体参考：[What about OpenCL and CUDA C++ alternatives? ](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives)
（3）OpenACC：For Open Accelerators，由 OpenACC 组织于 2011 年推出的 CPU + GPU 异构编程框架。不同于 CUDA 和 OpenCL 直接编写 GPU 代码，OpenACC 通过在代码中插入特殊的编译指示，依赖编译器实现异构并行加速，是一种类似 OpenMP 的编译指导语句式的编程技术。优点也是操作简单，可移植性强，缺点也是不灵活。目前主要是科研人员用于科学计算，专业程序员用的不多。
（4）其他更小众一点的：微软搞了 C++ AMP，AMD 搞了 ROCm，OpenMP 搞了 4.0 支持 GPU。
# 3 总结
一直以来，大多数开发者都采用面向 CPU 的方式进行学习和开发，也习惯了 CPU 顺序执行加有限并行的思维方式，这种知识也早已融入我们的教育中。但是，并行编程技术已经发展到 CPU + GPU 的异构并行编程时代，GPU 的开发已成为满足 AI 应用中并行处理需求的普遍做法，软件程序也从纯逻辑开发，变成逻辑 + AI 模型的时代。
因此，CPU 程序员应该学习 GPU 开发，更具体一点，如果你是一名 Linux 嵌入式开发，会使用 C/C++，那么你应该学习 GPU + CUDA C++，通过与硬件对话的思维方式进入 AI 技术的世界。








