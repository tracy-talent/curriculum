# 集群配置

------------------

配置好集群后，启动hadoop的hdfs，在hdfs上创建配置中用到的hdfs目录及文件

* 创建/app-logs，为yarn查看app运行日志使用
* 创建/history_log，为hadoop运行mapreduce程序查看历史日志用
* 创建/spark_event，存储spark运行过程中的事件日志
* 创建/spark/jars，将跑分布式程序所依赖的jar包上传到该目录下
* 创建/user/用户名，作为默认前缀目录，比如input相当于/user/用户名/input