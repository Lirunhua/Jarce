CC      = g++
CFLAGS  = -O3 -DNDEBUG
SOURCES = $(wildcard *.cpp)
# SOURCES定义了一个变量 wildcard为makefile内建函数，查找出当前目录下的所有cpp文件并赋值给SOURCES
OBJS := $(patsubst %.cpp, %.o,$(SOURCES))
# OBJS定义了一个变量 patsubst为makefile内建函数，把所有的.cpp都替换成.o，也就是把要产生的目标文件名都赋值给OBJS

petal:$(OBJS)
	@echo "源文件:" $(SOURCES)
	@echo "目标文件:" $(OBJS)
	$(CC) -o $@ $(OBJS) $(CFLAGS)
# petal依赖于所有的目标文件，当有目标文件被修改时，重新链接产生petal.exe

%.d :%.cpp
	@echo "create depend" $< $@;
	@set -e; \
	gcc -MM $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$
	
# @echo "create depend" $< $@; $< 是自动化变量 代表第一个依赖文件比如 kdbExt.cpp $@代表目标文件 kdbExt.d	
# @set -e; \   当后面的命令的返回值非0时，立即退出。
# gcc -MM $< > $@.$$$$; \   根据源文件产生依赖关系，并保存到以当前进程号为文件后缀名的临时文件中 
# sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \  sed是linux文本处理工具 从临时文件中读入依赖关系 将目标文件加入依赖关系的目录列表中，并保存到目标文件
# rm -f $@.$$$$ 删除临时文件


-include $(OBJS:.o=.d)
# 把所有的依赖文件.d文件都包含进来 $(OBJS:.o=.d)将OBJS变量里的.o都替换成了.d后缀
# 依赖文件描述了.o目标文件和.d依赖文件的依赖关系，一旦相应的源文件或者头文件有所修改，会重新产生依赖文件与目标文件


.PHONY:clean
        # .PHONY表示clean是一个伪目标 通过make clean(锤子和刷子)调用
clean:
	@echo "开始清理项目..."
	@echo "正在删除所有的.d文件"
	rm -f $(OBJS:.o=.d)
	@echo "正在删除所有的.o文件"
	rm -rf $(OBJS)
	@echo "正在删除petal"
	rm -f petal.exe
	@echo "清理结束"
