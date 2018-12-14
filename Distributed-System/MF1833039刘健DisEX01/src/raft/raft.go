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
import "sync/atomic"

//分别测试
//1.go test -v -run InitialElection
//2.go test -v -run ReElection

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

var heartbeatChannel = make([]chan bool, 3)

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

//
// A Go object implementing a single Raft peer.
//
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

func (rf *Raft) startElection() {
	rf.mu.Lock()
	if rf.state != Candidate {
		rf.mu.Unlock()
		return
	}
	
	args := RequestVoteArgs {
		Term: rf.currentTerm,
		CandidateId: rf.me,
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

type AppendEntriesArgs struct {}

type AppendEntriesReply struct {}

func (rf *Raft) AppendEntries(args AppendEntriesArgs, reply *AppendEntriesReply) {}
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


func (rf *Raft) sendHeatbeat() {
	rf.mu.lock()
	defer rf.mu.Unlock()
	//判断rf是否已失去连接
	if !rf.peers[rf.me].Call("Raft.AppendEntries", AppendEntriesArgs{}, &AppendEntriesReply{}) {
		DPrintf("%v crashed",rf.me)
		rf.state = Follower
		return
	}
	//向其它server发送心跳
	for i := 0; i < len(rf.peers); i++ {
		if i != rf.me {
			writeToChannel(heartbeatChannel[i]);
		}
	}
} 

func (rf *Raft) convertToFollower(term int) {
	rf.state = Follower
	rf.currentTerm = term
	rf.votedFor = VoteNull
}

func (rf *Raft) convertToCandidate() {
	rf.state = Candidate
	rf.currentTerm++
	rf.votedFor = rf.me
}

func (rf *Raft) convertToLeader() {
	rf.state = Leader
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
	return -1, -1, false
}

//
// the tester calls Kill() when a Raft instance won't
// be needed again. you are not required to do anything
// in Kill(), but it might be convenient to (for example)
// turn off debug output from this instance.
//
func (rf *Raft) Kill() {
	// Your code here, if desired.
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
	rf.currentTerm = 0
    rf.votedFor = VoteNull
	rf.state = Follower

	rf.grantVoteChannel = make(chan bool, 1)
	rf.leaderChannel = make(chan bool, 1)
    
    //heartbeat of leader
    heartbeatInterval := time.Duration(HeartbeatInterval)
	heartbeatChannel[me] = make(chan bool, 1)

    go func() {
        for {
            electionTimeout := getRandomElectionTimeout()
            rf.mu.Lock()
			state := rf.state
			DPrintf("Server(%d) state:%v, term:%v, electionTimeout:%v", rf.me, state, rf.currentTerm, electionTimeout)
            rf.mu.Unlock()

            switch state {
            case Follower:
                select {
				case <- heartbeatChannel[me]:
                case <- rf.grantVoteChannel:
                case <- time.After(electionTimeout):
					rf.mu.Lock()
                    rf.convertToCandidate()
                    rf.mu.Unlock()
                }
            case Candidate:
                go rf.startElection()
                select {
				case <- heartbeatChannel[me]:
                case <- rf.grantVoteChannel:
				case <- rf.leaderChannel:
                case <- time.After(electionTimeout):
					rf.mu.Lock()
                    rf.convertToCandidate()
                    rf.mu.Unlock()
                }
			case Leader:
				go rf.sendHeatbeat()
                time.Sleep(heartbeatInterval)
            } 
        }
    }()
	return rf
}
