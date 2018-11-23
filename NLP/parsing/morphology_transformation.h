#ifndef MORPHOLOGY_TRANSFORMATION_H
#define MORPHOLOGY_TRANSFORMATION_H

extern std::map<std::string, std::string> nounmap, verbmap, word2translation;

inline void split(std::string s, std::string t[], int &len){
    int cnt = 0;
    while (s[cnt] != '\n') {
        t[len] = "";
        while (s[cnt] != '\t' && s[cnt] != '\n')
            t[len] += s[cnt++];
        len++;
        if (s[cnt] == '\t') cnt++;
    }
    return;
}

void loadDic();

void loadIrNoun();

void loadIrVerb();

bool regularTrans(std::string, std::string&, std::map<std::string, std::string>);

#endif
