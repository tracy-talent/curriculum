#include <bits/stdc++.h>
#include "morphology_transformation.h"
using namespace std;
/**
 * 加入词形还原模块
*/
typedef pair<int, int> P;
const int RULELIMIT = 100;
const int WORDLIMIT = 1000;
const int UNACELIMIT = 1000; //unActiveEdge limit

struct activeEdge {
    int ruleno;  //规则映射编号
    int startno;  //始节点编号
    string symbol;  //当前归约到的符号
    activeEdge(int _ruleno = 0, int _startno = 0, string _symbol = ""):ruleno(_ruleno),startno(_startno),symbol(_symbol){}
};

struct unactiveEdge {
    int ruleno;  //规则映射编号
    P p;    //始、终节点编号
    string symbol;  //规则符号
    unactiveEdge(int _ruleno = 0, P _p = P(0,0), string _symbol = ""):ruleno(_ruleno),p(_p),symbol(_symbol){}
};

set<string> symbolset;
map<string, string> word2postag;
vector<string> rule_vec[RULELIMIT];  //存储规则
string rulemap[RULELIMIT];  //映射规则
string word[WORDLIMIT], wordmem[WORDLIMIT];
vector<activeEdge> ace_vec[WORDLIMIT];  //存储活动边
unactiveEdge chart_vec[UNACELIMIT];  //存储非活动边即结果
unactiveEdge agenda_que[UNACELIMIT];
int rear;
int rulenum = 0, wordnum;

inline void trim(string &str) {
    str.erase(str.find_last_not_of(" ") + 1);
    str.erase(0, str.find_first_not_of(" "));
    return;
}

inline string toUpperCase(string str) {
    for (int i = 0; i < str.length(); i++)
        if (str[i] >= 'a' && str[i] <= 'z')
            str[i] -= 32;
    return str;
}

void chart_loadDic() {
    /**词性
     * string pos[] = {"none.", "n.", "vt.", "vi.", "adj.", "adv.", "art.", "pron.",
     * "num.", "prep.", "conj.", "interj.", "vbl."};  //词性,interj.为感叹词 
    */
    FILE *fp = fopen("dic_ec.txt", "r");
    char line[500];
    string t[15];
    int len;
    while (fgets(line, 500, fp) != NULL) {
        len = 0;
        split(string(line), t, len);
        if (len > 2) {
            for (int i = 1; i < len - 2; i += 2)
                word2postag[t[0]] += toUpperCase(t[i].substr(0, t[i].length() - 1)) + "\t";
            word2postag[t[0]] += toUpperCase(t[len - 2].substr(0, t[len - 2].length() - 1)) + "\n";
        }
    }
    fclose(fp);
    loadDic();
    loadIrNoun();
    loadIrVerb();
    return;
}

inline void segment(string sentence) {
    trim(sentence);
    wordnum = 0;
    int space;
    while (1) {
        space = sentence.find_first_of(" ");
        if (space != -1) {
            wordmem[wordnum++] = sentence.substr(0, space);
        } else {
            wordmem[wordnum++] = sentence;
            break;
        }
        sentence = sentence.substr(space + 1);
        trim(sentence);
    }
    return;
}

bool isWordInDic() {
    string res;
    for (int i = 0; i < wordnum; i++) {
        if (word2postag.find(wordmem[i]) == word2postag.end()) {
            if (regularTrans(wordmem[i], res, word2postag)) {
                word[i] = res.substr(0, res.find("\t"));
            } else if (nounmap.find(wordmem[i]) != nounmap.end() && 
                        word2postag.find(nounmap[wordmem[i]]) != word2postag.end()) {
                word[i] = nounmap[wordmem[i]];
            } else if (verbmap.find(wordmem[i]) != verbmap.end() && 
                        word2postag.find(verbmap[wordmem[i]]) != word2postag.end()) {
                word[i] = verbmap[wordmem[i]];
            } else {
                return false;
            }
        } else {
            word[i] = wordmem[i];
        }
    }
    return true;
}

bool POSDetect(string sw[], int &lensw) {
    string t;
    bool flag = true;
    lensw = 0;
    for (int i = 0; i < wordnum; i++) {
        t = word2postag[word[i]];
        if (t.find("\t") == string::npos && t.substr(0,t.find("\n")) == "NONE") {
            sw[lensw++] = word[i];
            flag = false;
        }
    }
    return flag;
}

bool isSymbolInSet(string sw[], int &lensw) {
    string t[10];
    int len;
    bool flag = true;
    lensw = 0;
    for (int i = 0; i < wordnum; i++) {
        len = 0;
        split(word2postag[word[i]], t, len);
        bool f = false;
        for (int j = 0; j < len; j++) {
            if (t[j][0] == 'V') t[j] = "V";  //动词变换形式
            if (symbolset.find(t[j]) != symbolset.end())
                f = true;
        }
        flag = flag && f;
        if (!f) sw[lensw++] = word[i];
    }
    return flag;
}

bool judgeRule(string rule) {
    if (rule.find("->") != string::npos &&
    rule.substr(0,rule.find("->")).length() != 0 &&
    rule.substr(rule.find("->") + 2).length() != 0)
        return true;
    else
        return false;
}

void addRule(string srule) {
    rulemap[rulenum] = toUpperCase(srule.substr(0, srule.find_first_of("->")));
    trim(rulemap[rulenum]);
    symbolset.insert(rulemap[rulenum]);
    string t = toUpperCase(srule.substr(srule.find_first_of("->") + 2));
    trim(t);
    while (1) {
        int space = t.find_first_of(" ");
        if (space != -1) {
            rule_vec[rulenum].push_back(t.substr(0, space));
            symbolset.insert(rule_vec[rulenum][rule_vec[rulenum].size() - 1]);
        } else {
            rule_vec[rulenum].push_back(t);
            symbolset.insert(rule_vec[rulenum][rule_vec[rulenum].size() - 1]);
            break;
        }
        t = t.substr(space + 1);
        trim(t);
    }
    rulenum++;
    return;
}

void recoverAceVec(int veclen[]) {
    for (int i = 2; i <= wordnum + 1; i++){
        int k = ace_vec[i].size() - veclen[i];
        for (int j = 0; j < k; j++)
            ace_vec[i].pop_back();
    }
    return;
}

void matchRule(unactiveEdge uace, int &r) {
    for (int i = 0; i < rulenum; i++)
        if (rule_vec[i][0] == uace.symbol) {
            if (rule_vec[i].size() == 1) {
                agenda_que[++r] = unactiveEdge(i, uace.p, rulemap[i]);
            } else {
                ace_vec[uace.p.second].push_back(activeEdge(i, uace.p.first, uace.symbol));
            }
        }
    return;
}


string getNextSymbol(int idx, string pre, string post) {
    int sz = rule_vec[idx].size();
    for(int i = 0; i < sz; i++)
        if (rule_vec[idx][i] == pre) {
            if (rule_vec[idx][i + 1] == post) {
                if (i + 1 == sz - 1) {
                    return "";
                } else {
                    return rule_vec[idx][i + 1];
                }
            } else {
                return "-1";
            }
        }
    return "-1";
}

void matchActiveEdge(unactiveEdge uace, int &r) {
    int idx = uace.p.first;
    int sz = ace_vec[idx].size();
    for (int i = 0; i < sz; i++) {
        string ns = getNextSymbol(ace_vec[idx][i].ruleno, ace_vec[idx][i].symbol, uace.symbol);
        if (ns != "-1") {
            if (ns.length() == 0) {
                agenda_que[++r] = unactiveEdge(ace_vec[idx][i].ruleno,
                P(ace_vec[idx][i].startno, uace.p.second), rulemap[ace_vec[idx][i].ruleno]);
            } else {
                ace_vec[uace.p.second].push_back(activeEdge(ace_vec[idx][i].ruleno,
                ace_vec[idx][i].startno, uace.symbol));
            }
        }
    }
    return;
}


bool parser(int f, int r, int pos) {
    int veclen[wordnum+2];
    for (int i = 2; i <= wordnum + 1; i++)
        veclen[i] = ace_vec[i].size();

    while (f != r) {
        unactiveEdge uace = agenda_que[++f];
        matchRule(uace, r);
        matchActiveEdge(uace, r);
    }

    if (pos == wordnum) {
        if(r != -1 && agenda_que[r].symbol == "S"){
            rear = r;
            return true;
        } else {
            recoverAceVec(veclen);
            return false;
        }
    } else {
        string tags[10];
        int len = 0;
        string ptg = word2postag[word[pos]];
        split(ptg, tags, len);
        for (int i = 0; i < len; i++) {
            if (tags[i] != "NONE"){
                agenda_que[r + 1] = unactiveEdge(-1, P(pos+1, pos+2), (tags[i][0] == 'V') ? "V" : tags[i]);
                if(parser(f, r + 1, pos + 1))   return true;
            }
        }
        recoverAceVec(veclen);
        return false;
    }
}


int main() {
    ios::sync_with_stdio(false);
    cout << "**********************主菜单**********************" << endl;
    cout << "1. 写入规则" << endl;
    cout << "2. 句法分析" << endl;
    cout << "q. 退出程序" << endl;
    cout << "**************************************************" << endl;
    string choice;
    string ruleinput;
    string sentence;
    string word_error[WORDLIMIT];
    clock_t startTime, endTime;
    int welen;
    chart_loadDic();

    while (1) {
        cout << "请选择菜单选项(1或2回车，q退出)：";
        cin >> choice;
        cin.get();  //读入\n防止干扰下面程序的读写
        if (choice == "1") {
            cout << "请输入规则(pre -> post，可一次输入多条，输入\"q!\"退出)：" << endl;
            while (1) {
                getline(cin ,ruleinput);
                if (ruleinput == "q!")   break;
                if (judgeRule(ruleinput)){
                    addRule(ruleinput);
                    cout << "规则写入成功！" << endl;
                } else {
                    cout << "规则不规范请检查后重新输入！" << endl;
                }
            }
        } else if (choice == "2") {
            cout << "请输入待分析的句子：" << endl;
            getline(cin, sentence);
            segment(sentence);
            welen = 0;
            if (!isWordInDic()) {
                cout << "failed!\n句子中存在词典中未登记的词，无法完成句法分析！" << endl;
            } else if (!POSDetect(word_error, welen))  {
                cout << "failed!\n下面这些词在词典dic_ec.txt中词性标注为none.无意义，请更正词典词性！" << endl;
                for (int i = 0; i < welen; i++)
                    cout << word_error[i] << endl;
            } else if(!isSymbolInSet(word_error, welen)) {
                cout << "failed!\n下面这些词的词性未包含在句法规则中，无法完成句法分析！" << endl;
                for (int i = 0; i < welen; i++)
                    cout << word_error[i] << "\t" << word2postag[word_error[i]];
            } else {
                if (!parser(-1, -1, 0)) {
                    cout << "failed!\n句型有误！" << endl;
                } else {
                    startTime = clock();
                    cout << "successed!\n句型正确，分析结果如下：" << endl;
                    for (int i = 0; i <= rear; i++) {
                        cout << (i+1) << ". symbol:" << agenda_que[i].symbol << "   scope:(" <<
                        agenda_que[i].p.first << ", " << agenda_que[i].p.second << ")" << endl;
                        int rid = agenda_que[i].ruleno;
                        cout << "对应规则: ";
                        if (rid != -1) {
                            cout << agenda_que[i].symbol << " ->";
                            for (int j = 0; j < rule_vec[rid].size(); j++)
                                cout << " " + rule_vec[rid][j];
                            cout << endl;
                        } else {
                            cout << "单个的词无规则" << endl;
                        }
                        cout << "对应句块:";
                        for (int j = agenda_que[i].p.first; j < agenda_que[i].p.second; j++)
                            cout << " " + wordmem[j - 1];
                        cout << endl;
                    }

                    for (int i = 2; i <= wordnum + 1; i++)
                        ace_vec[i].clear();
                    endTime = clock();
                    cout << "分析耗时：" << (double)(endTime - startTime)/CLOCKS_PER_SEC << "s" << endl;
                }
            }
        } else if (choice == "q") {
            cout << "*** Good Bye! 欢迎再次使用！***";
            break;
        } else {
            cout << "没有该选项，请重新输入！" << endl;
        }
    }
    return 0;
}
