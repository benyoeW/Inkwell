# torch profiller
## 采集流程
### sglang
服务端：
1. 开启sglang服务
2. 进行压测

客户端：
开启压测之后，在压测过程中采用以下指令
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/trace_file // 这是保存采集文件的目标路径，如果不设置一般会保存到/tmp
curl -X POST http://127.0.0.1:8000/start_profile  // 开启的开关
curl -X POST http://127.0.0.1:8000/stop_profile  // 关闭的开关
采集的文件一般以*.trace.json.gz命名，若多个卡则有多个该文件
注意：采集了10s左右，大概3个G大小
### vllm
【待施工】
## hta
官方参考 https://docs.pytorch.org/tutorials/beginner/hta_intro_tutorial.html
`from hta.trace_analysis import TraceAnalysis`
> 采集文件的文件夹的地址，一般多个文件都放在一个文件夹内，hta会自动遍历这些文件
`trace_dir = "/Users/xawei/wxx_workspace/profilling/target_file_dir"`  

`analyzer = TraceAnalysis(trace_dir=trace_dir)`

下面是一些可视化方法，运行之后会打开浏览器自动显示hta分析信息

> time_spent_df = analyzer.get_temporal_breakdown()
> idle_time_df = analyzer.get_idle_time_breakdown()
> kernel_type_metrics_df, kernel_metrics_df = analyzer.get_gpu_kernel_breakdown()
> overlap_df = analyzer.get_comm_comp_overlap()




# nsys
