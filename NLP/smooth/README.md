==任务要求==：利用NLP技术判断句子通顺与否

> 环境要求：
>
> 1. python 3.6
> 2. pyltp 0.2.1(匹配的ltp mode 3.4.0)
> 3. numpy 1.15
> 4. centos 7.4(linux)

满足上述环境要求后，由于pyltp所需的模型太大就没上传了，附上[百度云链接](https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=%2F)，下载3.4.0版本然后解压得到ltp_data_v3.4.0目录放到smooth/input目录下，确保ltp_data_v3.4.0目录下有cws.model和pos.model这两个文件，接着进入到smooth/src目录下，然后执行以下命令来运行程序：

```shell]
python smooth.py
```

大概一分钟之后程序运行结束，程序运行结果存放在smooth/output目录下的result.txt