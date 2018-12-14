<center><h1>Assignment2: Raft Log Consensus</h1></center>

<center>日期：2018/12/13</center>



## 一、实验分析与设计

### 1.1 Raft算法简介

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Raft 是一种为了管理复制日志的一致性算法。它跟multi-paxos作用相同，效率也相当，但是它的组织结构跟
Paxos不同，Paxos难于理解且实现的工程难度太大，而Raft 算法更加容易理解并且更加容易被使用来构建系统。
Raft还包括一个用于变更集群成员的新机制，它使用重叠大多数(overlapping majorities)来保证安全性。Raft算法
主要分为领导者选举和追加日志， 在任务一中我们已经按照Raft算法的选举机制实现了领导者选举，领导者选举除了从candidate中选出leader，还扮演着一个重要的角色就是同步各个server的term。本次实验将在上次实验的基础上继续完善Raft算法的日志追加模块，这样选出来的leader就能从客户端接收命令并追加到集群中其他的server上。

### 1.2 设计思路与程序分析

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;实验设计是在理解了Raft这篇论文的分布式设计思想后，然后基于论文中图2中描述的4个逻辑模块用Go语言进行实现。AppendEntries RPC是本次实验要实现的核心模块，它的模块图下图1.1所示，需要很好的结合论文来对这个模块中各个变量所担当的角色进行理解，然后细化瓦解这个模块，逐步实现。

![1544772550079](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544772550079.png)

<center>图1.1 AppendEntries RPC模块图</center>

* leader在向其它raft server追加LogEntry的时候需要将自己的一些属性打包作为参数传递给它们，因此需要按AppendEntries RPC模块里的要求建一个struct专门用于存储leader的一些属性和要追加的日志条目数组，其中包含leader的任期Term，leader要追加的LogEntries起始处的前一个索引位置PrevLogIndex及其对应的任期PrevLogTerm，leader已提交到状态机上的LogEntry索引标记位置LeaderCommit，leader的这个struct的代码设计如下：

  ```go
  //from figure2 in paper->AppendEntries RPC:Arguments
  type AppendEntriesArgs struct {
      Term         int
      PrevLogIndex int
      PrevLogTerm  int
      Entries      []LogEntry
      LeaderCommit int
  }
  ```

* 除了需要建立一个传递内容的struct参数，还需要建立一个struct用于接收对Follower追加LogEntry的返回结果，这个struct需要包含被追加LogEntry的server当前任期term，是否追加成功的状态标志success，以及追加失败时返回的冲突索引位置ConflictIndex以及冲突任期ConflictTerm，返回结果的struct代码设计如下：

  ```go
  //from figure2 in paper->AppendEntries RPC:Results
  type AppendEntriesReply struct {
      Term          int
      Success       bool
      //properties not emerge in figure2 of paper->AppendEntries RPC:Results
      //leader向其它erver添加条目时出现的冲突信息
      ConflictTerm  int
      ConflictIndex int
  }
  ```

  这里有一点需要注意的地方，由于这两个struct有在文件外使用，而Go语法规定在文件外使用的变量需要以大写字母开头，所以这两个struct中的变量都需要以大写字母开头，否则测试时会报错，我个人就在这个地方吃到了bug。

* 建立好需要传递的参数之后就可以开始对AppendEntries RPC的逻辑部分进行实现了。具体的日志追加逻辑在论文中已经描述得很清晰了，这里就不再赘述，下面给出我这一部分的代码实现，代码中有注释以及关键部分对应所对应到的AppendEntries RPC模块图中的细节之处

  ```go
  //AppendEntries RPC Handler
  //from figure2 in paper->AppendEntries RPC:Receiver implementation
  func (rf *Raft) AppendEntries(args AppendEntriesArgs, reply *AppendEntriesReply) {
      rf.mu.Lock()
      defer rf.mu.Unlock()
      defer rf.persist()
  
      //返回参数
      success := false
      conflictTerm := 0
      conflictIndex := 0
  
      // all servers
      //利用任期做逻辑时钟同步各个server任期
      if args.Term > rf.currentTerm {
          rf.convertToFollower(args.Term)
      }
  
      if args.Term == rf.currentTerm {
          rf.state = Follower
          writeToChannel(rf.appendEntryChannel)
  
          if args.PrevLogIndex > rf.getLastLogIndex() {
              conflictIndex = len(rf.log)
              conflictTerm = -1
          } else {
              rfprevLogTerm := rf.log[args.PrevLogIndex].Term
              if args.PrevLogTerm != rfprevLogTerm {
                  //如果任期不等，返回rf这个任期内第一个entry的index
                  conflictTerm = rfprevLogTerm
                  for i := 1; i < len(rf.log); i++ {
                      if rf.log[i].Term == conflictTerm {
                          conflictIndex = i
                          break
                      }
                  }
              }
              
              // log entry in entries starts from 1
              if args.PrevLogIndex == 0 || args.PrevLogTerm == rfprevLogTerm {
                  success = true
                  rf.log = append(rf.log[0:args.PrevLogIndex+1], args.Entries...)
                  // Appendentries RPC--Reciever implementation-5：set commitIndex = min(leaderCommit,index of last new entry)
                  if args.LeaderCommit > rf.commitIndex {
                      rf.commitIndex = min(args.LeaderCommit, rf.getLastLogIndex())
                  }
              }
          }
      }
  
      //应用提交过的日志
      rf.applyLogs()
  
      reply.Term = rf.currentTerm
      reply.Success = success
      reply.ConflictIndex = conflictIndex
      reply.ConflictTerm = conflictTerm
      return
  }
  ```

* 依据AppendEntries所返回的结果还需要分情况讨论来处理，如果返回来的任期比leader的任期大，则将leader回退到follwer状态，否则进入下一步判断，此时如果leader的state和term依然保持着RPC之前的值并且返回的Success为真则表明追加成功，更新leader的nextIndex[]和matchIndex[]，同时不要忘了依据更新之后的matchIndex[]的中位数及其任期与leader的commitIndex及leader的任期进行比较来决定是否对leader的commitIndex进行更新。如果返回的Success为假则更新leader对应到被追加LogEntry的server所对应的nextIndex，这样下次就可以以一个新的追加位置来进行匹配验证。AppendEntries返回结果的处理代码如下

  ```go
  rf.mu.Lock()
  if reply.Term > rf.currentTerm {
      rf.convertToFollower(reply.Term)
      rf.mu.Unlock()
      return
  }
  
  if rf.state != Leader || rf.currentTerm != args.Term {
      rf.mu.Unlock()
      return
  }
  
  if reply.Success {
      // AppendEntries成功，更新对应raft实例的nextIndex和matchIndex值
      rf.matchIndex[idx] = args.PrevLogIndex + len(args.Entries)
      rf.nextIndex[idx] = rf.matchIndex[idx] + 1
      rf.updateCommitIndex()
      rf.mu.Unlock()
      return
  } else {
      // AppendEntries失败，减小对应raft实例的nextIndex的值重试
      newIndex := reply.ConflictIndex
      for i := 1; i < len(rf.log); i++ {
          entry := rf.log[i]
          if entry.Term == reply.ConflictTerm {
              newIndex = i + 1  //nextIndex比preLogIndex大1
          }
      }
      rf.nextIndex[idx] = max(1, newIndex)
      rf.mu.Unlock()
  }
  ```

* 上面代码中对leader的commitIndex更新逻辑实现代码如下

  ```go
  //from figure2 in paper->Rules for servers:Leaders
  //当Log Entry发送到过半Server上时就更新leader的commitIndex提交这些entry，取matchIndex[]的中位数作为过半值参考
  func (rf *Raft) updateCommitIndex() {
      matchIndexCopy := make([]int, len(rf.matchIndex))
      copy(matchIndexCopy, rf.matchIndex)
      matchIndexCopy[rf.me] = len(rf.log) - 1
      sort.Ints(matchIndexCopy)
      
      N := matchIndexCopy[len(rf.peers)/2]
      if rf.state == Leader && N > rf.commitIndex && rf.log[N].Term == rf.currentTerm {
          //更新commitIndex，并且应用到状态机
          rf.commitIndex = N
          rf.applyLogs()
      }
  }
  ```

* 对于每个raft server上已提交的LogEntry(索引位置有commitIndex标识)最终还需要apply到状态机上，每个server都会在外部调用Start方法来对提交的日志条目进行应用，代码如下

  ```go
  func (rf *Raft) Start(command interface{}) (int, int, bool) {
  	//Your code here
  	rf.mu.Lock()
      defer rf.mu.Unlock()
  
      //返回值
      term := rf.currentTerm
      index := -1
      isleader := (rf.state == Leader)
  
      if isleader {
          index = rf.getLastLogIndex() + 1
          entry := LogEntry{
              Term:    term,
              Index:   index,
              Command: command,
          }
          rf.log = append(rf.log, entry)
          rf.persist()
      }
  
      return index, term, isleader
  }
  ```

* 综合前面实现的Leader Election，到这里Raft的两大核心模块的设计实现都已完成，现在需要写一个驱动模块来对这两个核心模块进行组合以及每个server的状态进行管控，这里有一个trick就是使用switch case和Go Channel来实现这个驱动模块，代码实现如下

  ```go
  switch state {
  case Follower:
      select {
      case <- rf.grantVoteChannel:
      case <- rf.appendEntryChannel:
      case <- time.After(electionTimeout):
          rf.mu.Lock()
          rf.convertToCandidate()
          rf.mu.Unlock()
      }
  case Candidate:
      go rf.startElection()
      select {
      case <- rf.grantVoteChannel:
      case <- rf.appendEntryChannel:
      case <- rf.leaderChannel:
      case <- time.After(electionTimeout):
          rf.mu.Lock()
          rf.convertToCandidate()
          rf.mu.Unlock()
      }
  case Leader:
      go rf.startAppendEntries()
      time.Sleep(heartbeatInterval)
  } 
  ```



## 二、实验结果

* TestBasicAgree的测试结果

  ![1544752098831](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544752098831.png)

* TestFailAgree的测试结果

  ![1544752208371](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544752208371.png)

* TestFailNoAgree的测试结果

  ![1544752278833](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544752278833.png)

* TestConcurrentStarts的测试结果

  ![1544752361055](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544752361055.png)

* TestRejoin的测试结果

  ![1544764920936](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544764920936.png)

* TestBackup的测试结果

  ![1544765044693](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544765044693.png)

* TestPersist1的测试结果

  ![1544765155382](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544765155382.png)

* TestPersist2的测试结果

  ![1544765199552](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544765199552.png)

* TestPersist3的测试结果

  ![1544765233922](C:\Users\LiuJian\AppData\Roaming\Typora\typora-user-images\1544765233922.png)


## 三、实验总结

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;相比领导者选举，日志追加无论是在理论上还是实现上，要考虑到的细节都要比领导者选举多。领导者选举模块只需比较Follower和Candidate的任期以及最后一个LogEntry。而日志追加模块先比较leader和其它server的任期，如果合理则比较它们的prevLogIndex，如果合理则接着比较两者在preLogIndex处的LogEntry所对应的任期，并根据日志任期是否相等来返回confilictTerm和conflictIndex。Leader对接收到的结果处理也要繁琐一些，如果追加失败还需要根据返回的conflictTerm和conflictIndex来更新leader的nextIndex，然后用nextIndex更新传递的参数preLogIndex和preLogTerm。在向其它server追加日志的时候可能会反复匹配失败从而leader需要反复更新传递参数进行反复匹配，直到匹配到一个合适的追加位置，这样才能保证其它server的LogEntry队列与Leader的LogEntry队列保持一致。这么多的细节，同时还要控制多线程对共享资源的访问，给实现增添了难度，很容易漏掉某个小细节而使程序出错，而多线程程序编写又不提供单步调试的机制，因此最好的调试方式就是把关键地方的信息print出来，通过这种方式帮助我解决了程序中出现的很多问题。而使用Go提供的一些并发原语比如select，go，chan，Mutex等等，让并发和异步处理上更为方便简洁，让我们得以集中在Raft算法实现上而不必花费太多心思去考虑处理异步线程。然后说一下测试部分，由于测试程序中有随机宕机的代码，所以测试的时候最好多测几次以保证程序的准确性。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最后综合前两次的实验作一个小总结，作为分布式一致性协议，保证一致性的前提下，在可理解性和实现难度上，Raft要比Paxos友好得多，论文中有一些非常有启发性的思想非常值得我们去思考和借鉴，我在理解和复现论文的过程中给我的最大启发有以下三点：

* 随机化每个server的选举时间来解决多个Candidate竞选Leader时因选票不足而无法产生Leader的问题
* 利用Leader Election的过程来同步各个server的任期
* 利用过半选举和过半提交的机制来保证系统的==安全性==(已提交的LogEntry不会被后面追加的LogEntry覆盖掉，最终保证状态机的一致性)