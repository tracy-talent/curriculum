package raft

//
// this is an outline of the API that raft must expose to
// the service (or tester). see comments below for
// each of these functions for more details.
//
// rf = Make(...)
//   create a new Raft server.
// rf.Start(command interface{}) (index, term, isleader)
//   start agreement on a new log entry
// rf.GetState() (term, isLeader)
//   ask a Raft for its current term, and whether it thinks it is leader
// ApplyMsg
//   each time a new entry is committed to the log, each Raft peer
//   should send an ApplyMsg to the service (or tester)
//   in the same server.
//

import "sync"
import "labrpc"
import "math/rand"
import "time"
import "sort"
import "bytes"
import "encoding/gob"
import "sync/atomic"

//constant variable use in raft.go
const (
	Follower = iota
	Candidate
	Leader

	HeartbeatInterval = 50
    ElectionTimeoutLower = 300
    ElectionTimeoutUpper = 400

    VoteNull = -1
)

//
// as each Raft peer becomes aware that successive log entries are
// committed, the peer should send an ApplyMsg to the service (or
// tester) on the same server, via the applyCh passed to Make().
//
type ApplyMsg struct {
	Index       int		//index of a LogEntry
	Command     interface{}		//the command contains in LogEntry
	UseSnapshot bool   // ignore for lab2; only used in lab3
	Snapshot    []byte // ignore for lab2; only used in lab3
}

//Leader will send LogEntry to other servers
type LogEntry struct {
	Index	int
	Term	int
	Command	interface{}
}

//
// A Go object implementing a single Raft peer.
//
//from figure 2:State in paper
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
	log			[]LogEntry  //LogEntry queue of the server:receive LogEntry from other server(first index is 1)
	
	//Volatile state on all servers
	commitIndex int
    lastApplied int

    //Volatile state on leaders
    nextIndex  []int
    matchIndex []int

	//selfdefinition metadata
	state   int

	//channel
	applyChannel       	chan ApplyMsg
    appendEntryChannel 	chan bool
	grantVoteChannel	chan bool
	leaderChannel		chan bool
	killChannel			chan bool
}

// return currentTerm and whether this server
// believes it is the leader.
func (rf *Raft) GetState() (int, bool) {
	var term int
	var isleader bool

	// Your code here.
	rf.mu.Lock()
	defer rf.mu.Unlock()
	term = rf.currentTerm
	isleader = (rf.state == Leader)
	return term, isleader
}

//
// save Raft's persistent state to stable storage,
// where it can later be retrieved after a crash and restart.
// see paper's Figure 2 for a description of what should be persistent.
//
func (rf *Raft) persist() {
	// Your code here.
	// Example:
	// w := new(bytes.Buffer)
	// e := gob.NewEncoder(w)
	// e.Encode(rf.xxx)
	// e.Encode(rf.yyy)
	// data := w.Bytes()
	// rf.persister.SaveRaftState(data)
	w := new(bytes.Buffer)
	e := gob.NewEncoder(w)
	e.Encode(rf.currentTerm)
	e.Encode(rf.votedFor)
	e.Encode(rf.log)
	data := w.Bytes()
	rf.persister.SaveRaftState(data)
}

//
// restore previously persisted state.
//
func (rf *Raft) readPersist(data []byte) {
	// Your code here.
	// Example:
	// r := bytes.NewBuffer(data)
	// d := gob.NewDecoder(r)
	// d.Decode(&rf.xxx)
	// d.Decode(&rf.yyy)
	rf.mu.Lock()
	defer rf.mu.Unlock()
	if data != nil && len(data) > 0 {
		r := bytes.NewBuffer(data)
		d := gob.NewDecoder(r)
		d.Decode(&rf.currentTerm)
		d.Decode(&rf.votedFor)
		d.Decode(&rf.log)
	}
}


//
// example RequestVote RPC arguments structure.
//
//from figure2: RequestVote RPC in paper
type RequestVoteArgs struct {
	// Your data here.
	Term			int	//candidate's term
	CandidateId		int	//candidate requesting vote
	LastLogIndex	int //index of cadadate's last log entry
	LastLogTerm		int //term of cadidate's last log entry
}

//
// example RequestVote RPC reply structure.
//
//from figure2: RequestVote RPC in paper
type RequestVoteReply struct {
	// Your data here.
	Term		int //currentTerm, for candidate's last log entry
	VoteGranted	bool //true means candidate received vote 
}

//
// example RequestVote RPC handler.
//
func (rf *Raft) RequestVote(args RequestVoteArgs, reply *RequestVoteReply) {
	// Your code here.
	rf.mu.Lock()
    defer rf.mu.Unlock()
    defer rf.persist()

    //make term of different server be consitent(so vote also work as Logic Clock)
	if (rf.currentTerm < args.Term) {
		rf.convertToFollower(args.Term)
	}

	//according to figure2 in paper->RequestVote RPC:Receiver implementation
	voteGranted := false
	if rf.currentTerm == args.Term && (rf.votedFor == VoteNull || rf.votedFor == args.CandidateId) && (rf.getLastLogTerm() < args.LastLogTerm || (rf.getLastLogTerm() == args.LastLogTerm && rf.getLastLogIndex() <= args.LastLogIndex)) {
		voteGranted = true
		writeToChannel(rf.grantVoteChannel)
		rf.votedFor = args.CandidateId
		rf.state = Follower
	}

	reply.Term = rf.currentTerm
	reply.VoteGranted = voteGranted
}

func (rf *Raft) startElection() {
	rf.mu.Lock()
	if rf.state != Candidate {
		rf.mu.Unlock()
		return
	}
	
	args := RequestVoteArgs {
		Term: rf.currentTerm,
		CandidateId: rf.me,
		LastLogIndex: rf.getLastLogIndex(),
		LastLogTerm: rf.getLastLogTerm(),
	}
	rf.mu.Unlock()

	var votedNum uint32 = 1  //rf得到的总票数
	for i:= 0; i < len(rf.peers); i++ {
		if i == rf.me {
			continue
		}
		
		rf.mu.Lock()
		if rf.state != Candidate || rf.currentTerm != args.Term {
			rf.mu.Unlock()
			return
		}
		rf.mu.Unlock()
		go func(idx int) {
			reply := &RequestVoteReply{}
			retv := rf.sendRequestVote(idx, args, reply)
			if retv {
				rf.mu.Lock()
				defer rf.mu.Unlock()
				if reply.Term > rf.currentTerm {
					rf.convertToFollower(reply.Term)
					return
				}

				if rf.state != Candidate || rf.currentTerm != args.Term {
					return
				}

				if reply.VoteGranted {
					atomic.AddUint32(&votedNum, uint32(1));
				}

				if atomic.LoadUint32(&votedNum) > uint32(len(rf.peers) / 2) {
					rf.convertToLeader()
					writeToChannel(rf.leaderChannel)
				}
			}
		}(i)
	}
}

//from figure2 in paper->AppendEntries RPC:Arguments
type AppendEntriesArgs struct {
    Term         int
    PrevLogIndex int
    PrevLogTerm  int
    Entries      []LogEntry
    LeaderCommit int
}

//from figure2 in paper->AppendEntries RPC:Results
type AppendEntriesReply struct {
    Term          int
    Success       bool
    //properties not emerge in figure2 of paper->AppendEntries RPC:Results
    //leader向其它erver添加条目时出现的冲突信息
    ConflictTerm  int
    ConflictIndex int
}

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

func (rf *Raft) startAppendEntries() {
    for i := 0; i < len(rf.peers); i++ {
        if i == rf.me {
            continue
        }

        go func(idx int) {
            for {
                rf.mu.Lock()
                if rf.state != Leader {
                    rf.mu.Unlock()
                    return
                }

                nextIndex := rf.nextIndex[idx]
                entries := make([]LogEntry, 0)
                entries = append(entries, rf.log[nextIndex:]...)
                args := AppendEntriesArgs{
                    Term:         rf.currentTerm,
                    PrevLogIndex: rf.getPrevLogIndex(idx),
                    PrevLogTerm:  rf.getPrevLogTerm(idx),
                    Entries:      entries,
                    LeaderCommit: rf.commitIndex,
                }
                reply := &AppendEntriesReply{}
                rf.mu.Unlock()

                retv := rf.sendAppendEntries(idx, args, reply)
                if retv {
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
	            }    
            }
        }(i)
    }
}
//
// example code to send a RequestVote RPC to a server.
// server is the index of the target server in rf.peers[].
// expects RPC arguments in args.
// fills in *reply with RPC reply, so caller should
// pass &reply.
// the types of the args and reply passed to Call() must be
// the same as the types of the arguments declared in the
// handler function (including whether they are pointers).
//
// returns true if labrpc says the RPC was delivered.
//
// if you're having trouble getting RPC to work, check that you've
// capitalized all field names in structs passed over RPC, and
// that the caller passes the address of the reply struct with &, not
// the struct itself.
//
func (rf *Raft) sendRequestVote(server int, args RequestVoteArgs, reply *RequestVoteReply) bool {
	ok := rf.peers[server].Call("Raft.RequestVote", args, reply)
	return ok
}

func (rf *Raft) sendAppendEntries(server int, args AppendEntriesArgs, reply *AppendEntriesReply) bool {
    ok := rf.peers[server].Call("Raft.AppendEntries", args, reply)
    return ok
}

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

func (rf *Raft) getPrevLogIndex(idx int) int {
    return rf.nextIndex[idx] - 1
}

func (rf *Raft) getPrevLogTerm(idx int) int {
    prevLogIndex := rf.getPrevLogIndex(idx)
    if prevLogIndex == 0 {
        return -1
    } else {
        return rf.log[prevLogIndex].Term
    }
}

func (rf *Raft) getLastLogIndex() int {
	return len(rf.log) - 1 
}

func (rf *Raft) getLastLogTerm() int {
	lastLogIndex := rf.getLastLogIndex()
	//log index starts from 1
	if lastLogIndex == 0 {
		return -1
	}
	return rf.log[lastLogIndex].Term
}

func (rf *Raft) convertToFollower(term int) {
	defer rf.persist()
	rf.state = Follower
	rf.currentTerm = term
	rf.votedFor = VoteNull
}

func (rf *Raft) convertToCandidate() {
	defer rf.persist()
	rf.state = Candidate
	rf.currentTerm++
	rf.votedFor = rf.me
}

// 转换成leader时记得更新Leader特有的nextIndex和matchIndex
func (rf *Raft) convertToLeader() {
	defer rf.persist()
	rf.state = Leader
	for i := 0; i < len(rf.peers); i++ {
		rf.nextIndex[i] = rf.getLastLogIndex() + 1
		rf.matchIndex[i] = 0
	}
}

func writeToChannel(channel chan bool) {
	select {
	case <- channel:
	default:
	}
	channel <- true
}

func getRandomElectionTimeout() time.Duration {
    return time.Millisecond * time.Duration(rand.Intn(ElectionTimeoutUpper - ElectionTimeoutLower) + ElectionTimeoutLower)
}


// 对于所有服务器都需要执行的，在AppendEntry中更新commitIndex后调用
// applyCh在test_test.go中要用到
// 应用日志条目到状态机，from figure2 in paper->Rules For Servers:All Servers
func (rf *Raft) applyLogs() {
    for rf.commitIndex > rf.lastApplied {
        rf.lastApplied++
        entry := rf.log[rf.lastApplied]
        apm := ApplyMsg{
            Index:   entry.Index,
            Command: entry.Command,
        }
        rf.applyChannel <- apm
    }
}
//
// the service using Raft (e.g. a k/v server) wants to start
// agreement on the next command to be appended to Raft's log. if this
// server isn't the leader, returns false. otherwise start the
// agreement and return immediately. there is no guarantee that this
// command will ever be committed to the Raft log, since the leader
// may fail or lose an election.
//
// the first return value is the index that the command will appear at
// if it's ever committed. the second return value is the current
// term. the third return value is true if this server believes it is
// the leader.
//
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

//
// the tester calls Kill() when a Raft instance won't
// be needed again. you are not required to do anything
// in Kill(), but it might be convenient to (for example)
// turn off debug output from this instance.
//
func (rf *Raft) Kill() {
	// Your code here, if desired.
	writeToChannel(rf.killChannel)
}

//
// the service or tester wants to create a Raft server. the ports
// of all the Raft servers (including this one) are in peers[]. this
// server's port is peers[me]. all the servers' peers[] arrays
// have the same order. persister is a place for this server to
// save its persistent state, and also initially holds the most
// recent saved state, if any. applyCh is a channel on which the
// tester or service expects Raft to send ApplyMsg messages.
// Make() must return quickly, so it should start goroutines
// for any long-running work.
//
func Make(peers []*labrpc.ClientEnd, me int,
	persister *Persister, applyCh chan ApplyMsg) *Raft {
	rf := &Raft{}
	rf.peers = peers
	rf.persister = persister
	rf.me = me

	// Your initialization code here.
	//initiate Raft state
	rf.currentTerm = 0
    rf.votedFor = VoteNull
	rf.log = append(make([]LogEntry, 0), LogEntry{})
	rf.state = Follower
	rf.commitIndex = 0
    rf.lastApplied = 0
    rf.nextIndex = make([]int, len(peers))
    rf.matchIndex = make([]int, len(peers))

    rf.applyChannel = applyCh
	rf.grantVoteChannel = make(chan bool, 1)
	rf.appendEntryChannel = make(chan bool, 1)
	rf.leaderChannel = make(chan bool, 1)
    rf.killChannel = make(chan bool, 1)
    
    //get random heartbeatInterval
    heartbeatInterval := time.Duration(HeartbeatInterval)

	// initialize from state persisted before a crash
	rf.readPersist(persister.ReadRaftState())

    go func() {
    Loop:
        for {
            select {
            case <- rf.killChannel:
                break Loop
            default:
            }

            electionTimeout := getRandomElectionTimeout()
            rf.mu.Lock()
            state := rf.state
            rf.mu.Unlock()

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
        }
    }()
	return rf
}