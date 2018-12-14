<center><h1>Assignment 1: Raft Leader Election</h1></center>

<center>日期：2018/11/27</center>



## 一、实验分析与设计

## 1.1 算法简介

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Raft 是一种为了管理复制日志的一致性算法。它跟multi-paxos作用相同，效率也相当，但是它的组织结构跟Paxos不同，Paxos难于理解且实现的工程难度太大，而Raft 算法更加容易理解并且更加容易被使用来构建系统。 Raft还包括一个用于变更集群成员的新机制，它使用重叠大多数(overlapping majorities)来保证安全性。Raft算法主要分为领导者选举和追加日志，任务1主要是对领导者选举进行理解与实现。下面是论文中给出的Raft算法的浓缩图，程序的设计是在理解Raft算法的基础上然后围绕这张图来实现整个算法。

![1543331189665](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1543331189665.png)

<center>图1.1  Raft算法概览图</center>

## 1.2 设计思路与程序分析

* Raft算法存在三种状态分别为leader,follower,candidate，下图1.2是这三者之间的状态转换图：

  ![1543331645647](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1543331645647.png)

  <center>图1.2  Raft的状态转换图</center>

  在程序中设置静态变量来标识状态，下面是程序中的一些静态变量，iota从零开始枚举。后面是分布式系统中常见的心跳和时钟，HeartbeatInterval是leader发送的心跳间隔，ElectionTimeoutLower是时钟的下确界，ElectionTimeoutUpper是时钟的上确界。

  ```go
  const (
  	Follower = iota
  	Candidate
  	Leader
  
  	HeartbeatInterval = 50
      ElectionTimeoutLower = 300
      ElectionTimeoutUpper = 400
  
      VoteNull = -1
  )
  ```

* Raft算法的时钟主要有两个功能，一是用于控制状态转换，而是控制投票选举。当一个follower超时未收到leader的消息则转换成candidate并发起投票，只有获得半成票以上的才有资格当选leader，因此为了不让各个candidate平分票而陷入无意义的循环，需要对各个server的时钟间隔做一个随机化处理：

  ```go
  func getRandomElectionTimeout() time.Duration {
      return time.Millisecond * time.Duration(rand.Intn(ElectionTimeoutUpper - ElectionTimeoutLower) + ElectionTimeoutLower)
  }
  ```

* 每个server存储着图1.1中State那一部分的状态，在程序中建一个struct来存储这些状态：

  ```go
  type Raft struct {
  	mu        sync.Mutex	//add lock before access a Raft
  	peers     []*labrpc.ClientEnd  //ports used to connect other servers
  	persister *Persister	//stable storage of Raft state
  	me        int // index into peers[]
  
  	// Your data here.
  	// Look at the paper's Figure 2 for a description of what
  	// state a Raft server must maintain.
  	currentTerm	int  //the term of a election(任期)
  	votedFor	int	//candidateId the server votes for,initialized by -1
  
  	//selfdefinition metadata
  	state   int
  
  	//channel
  	grantVoteChannel	chan bool
  	leaderChannel		chan bool
  }
  ```

* 有了各个server的状态之后，就可以在RequestVote模块中利用这些状态进行判断和处理，图1.1中Request RPC模块中Arguments有4个，Results有两个，但是任务一并没有涉及添加日志条目，所以并不需要这么多的参数和返回结果，Arguments和Results都只需要term任其号即可，因为在不加入日志模块的情况下，Raft的领导选举仅靠term任期号即可实现，下面是RequestVoteArgs和RequstVoteReply的Struct代码，由于需要rpc外部访问，struct的内部变量名一定要以大写字母开头：

  ```go
  //
  // example RequestVote RPC arguments structure.
  //
  type RequestVoteArgs struct {
  	// Your data here.
  	Term			int	//candidate's term
  	CandidateId		int	//candidate requesting vote
  }
  
  //
  // example RequestVote RPC reply structure.
  //
  type RequestVoteReply struct {
  	// Your data here.
  	Term		int //currentTerm, for candidate's last log entry
  	VoteGranted	bool //true means candidate received vote 
  }
  ```

* leader election最为核心的模块就是RequestVote了，通过模拟rpc调用向其它server请求投票，记得每次访问一个raft server的状态时先用mutex对其加锁进行独占访问，如果一个server的任期号比发起请求的candidate小就将该server转换成follower，如果比较他们任期term，如果相等，并且这个server还没有投票给其他的candidate则把票投给这个candidate，然后返回这个server的任期term。

  ```go
  //
  // example RequestVote RPC handler.
  //
  func (rf *Raft) RequestVote(args RequestVoteArgs, reply *RequestVoteReply) {
  	// Your code here.
  	rf.mu.Lock()
      defer rf.mu.Unlock()
  
      //make term of different server be consitent(so vote also work as Logic Clock)
  	if (rf.currentTerm < args.Term) {
  		rf.convertToFollower(args.Term)
  	}
  
  	voteGranted := false
  	if rf.currentTerm == args.Term && (rf.votedFor == VoteNull || rf.votedFor == args.CandidateId) {
  		voteGranted = true
  		writeToChannel(rf.grantVoteChannel)
  		rf.votedFor = args.CandidateId
  		rf.state = Follower
  	}
  
  	reply.Term = rf.currentTerm
  	reply.VoteGranted = voteGranted
  }
  ```

* 对于返回的任期号，如果比candidate的任期号大，则candidate转换成follower，退出选举。

  ```go
  if reply.Term > rf.currentTerm {
  					rf.convertToFollower(reply.Term)
  					return
  				}
  ```

## 二、实验总结

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;虽然 Raft 协议相对其他容错分布式一致性协议来说更容易理解，但是在实现和调试过程中也遇到不少细节问题。 Go提供了很多方便好用的并发原语比如：select,go,chan,Mutex等等，因此Raft 使用 Go 语言处理的优势是并发和异步处理上非常简洁明了，不用费心思去处理异步线程。
通过这次作业我理清了 Raft 的大体逻辑

* 发现小的 term 丢弃

* 发现大的 term，跟新自身 term，转换为 Follower， 重置投票

*  代码中还有 Leader/Follower/Candidate 的 Heartbeat routine 逻辑

  目前任务一这一次的实验还挺简单，任务二可能会比较有挑战性一点，我会更深一步研读论文和学习Go的开发。

结果截图：

![1544765804602](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544765804602.png)