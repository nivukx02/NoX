#include "synchronization/audio_event_sync.h"
#include <algorithm>
#include <cmath>
#include <limits>
namespace nox::native::sync {
AudioSyncResult AudioEventSync::estimate(const std::vector<double>&a,const std::vector<double>&b)const{
    AudioSyncResult o;if(a.size()<16||b.size()<16)return o;double best=-2;int lagBest=0;
    for(int lag=-static_cast<int>(b.size())+1;lag<static_cast<int>(a.size());++lag){double dot=0,aa=0,bb=0;for(std::size_t i=0;i<a.size();++i){long long j=static_cast<long long>(i)+lag;if(j<0||j>=static_cast<long long>(b.size()))continue;const double x=a[i],y=b[static_cast<std::size_t>(j)];dot+=x*y;aa+=x*x;bb+=y*y;}const double d=std::sqrt(aa*bb);if(d>1e-12){double c=dot/d;if(c>best){best=c;lagBest=lag;}}}
    o.offsetSamples=lagBest;o.correlation=std::max(-1.0,std::min(1.0,best));o.confidence=std::clamp((o.correlation-.5)/.5,0.0,1.0);o.valid=o.correlation>.7;return o;
}
}