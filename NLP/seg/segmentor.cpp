#include <cstdio>
#include <cstring>
#include <map>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;
map<string, bool> wordmap;

void loadDic() {
    FILE *fp = fopen("dic_ce.txt", "r");
    char line[200];
    string str;
    while(fgets(line, 200, fp) != NULL) {
        str = string(line);
        wordmap[str.substr(0, str.find_first_of(","))] = true;
    }
    fclose(fp);
    return;
}

void segment(vector<string> &seg, string st) {
    int pos = 0;
    int sz = st.length();
    string t;
    int cnt = 0, spos;
    while (pos < sz) {
        cnt = pos;
        spos = pos;
        t = "";
        while (st[cnt]) {
            t += st.substr(cnt, 2);
            if (wordmap.find(t) != wordmap.end())
                pos = cnt + 2;
            cnt += 2;
        }
        if (pos == spos) {
            seg.push_back(st.substr(spos, 2));
            pos += 2;
        }else {
            seg.push_back(st.substr(spos, pos - spos));
        }
    }
    return;
}

bool inputJudge(string str) {
    int sz = str.length();
    string symbol = ",.!?";
    for (int i = 0; i < sz; i++)
        if ((str[i] >= '0' && str[i] <= '9') ||
            (str[i] >= 'a' && str[i] <= 'z') ||
            (str[i] >= 'A' && str[i] <= 'Z') ||
            symbol.find(str[i]) != string::npos) {
                return false;
            }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    loadDic();
    string punc = "，。！？";
    string str, sentence;
    clock_t startTime, endTime;
    vector<string> vec, seg;
    int cnt;
    while(1) {
        cout << "请输入句子(只支持中文，支持标点分隔的多句输入): ";
        getline(cin, sentence);
        if (sentence.length() == 0 || !inputJudge(sentence)) {
            cout << "输入不符合规范，请检查后重新输入!" << endl;
            continue;
        }
        startTime = clock();
        cnt = 0;
        str = "";
        while (sentence[cnt]) {
            if (sentence[cnt] != ' ') {
                if (punc.find(sentence.substr(cnt, 2)) == string::npos)
                    str += sentence.substr(cnt, 2);
                else{
                    seg.clear();
                    segment(seg, str);
                    vec.insert(vec.end(), seg.begin(), seg.end());
                    vec.push_back(sentence.substr(cnt, 2));
                    str = "";
                }
            }
            cnt += 2;
        }
        if (str.length() > 0) {
            seg.clear();
            segment(seg, str);
            vec.insert(vec.end(), seg.begin(), seg.end());
        }
        for (int i = 0; i < vec.size() - 1; i++)
            cout << vec[i] << "/";
        if(vec.size() > 0)
            cout << vec[vec.size() - 1] << endl;
        else
            cout << "分词失败，请确认输入是否符合规则。" << endl;
        vec.clear();
        endTime = clock();
        cout << "分词耗时：" << (double)(endTime - startTime)/CLOCKS_PER_SEC << "s" << endl;
    }
    return 0;
}
