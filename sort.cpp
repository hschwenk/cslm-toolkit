#include<vector>
#include<math.h>
#include <iostream>
#include <algorithm>

//simple exponential decay as length penalty (input length = output length: no penalty)
REAL weight_lengths(std::size_t input_length, std::size_t output_length) {
    return log(0.8)*abs(input_length-output_length);
}


//change data structure (vector of vectors of pairs) and prune number of hypotheses per length to N
std::vector<std::vector<std::pair<REAL, std::size_t> > > prepare_hypotheses(REAL* scores, std::size_t maxLength, std::size_t vocab_size, std::size_t Nbest) {

    // outermost vector: one item per length
    std::vector<std::vector<std::pair<REAL, std::size_t> > > ret;

    // for each length
    for(std::size_t i = 0; i < maxLength; ++i){
        std::vector<std::pair<REAL, std::size_t> > vec (vocab_size);

        // for each word in the vocabulary
        for(std::size_t j = (i*vocab_size); j < ((i+1)*vocab_size); ++j){
            std::size_t idx = j-(i*vocab_size);
            vec[idx] = std::make_pair(scores[j],idx); //store probability and index
        }

        // prune to N most probable members
        std::nth_element(vec.begin(), min(vec.end(),vec.begin()+Nbest), vec.end(), std::greater<std::pair<REAL, std::size_t> >());
        vec.resize(std::min(Nbest,vec.size()));

        ret.push_back(vec);
    }
    return ret;
}

std::vector<std::pair<REAL, std::vector<std::size_t> > > sort_ngrams(std::vector<std::vector<std::pair<REAL, std::size_t> > > scores, std::size_t input_length, std::size_t Nbest) {

    //stack of hypotheses for building next greater length
    std::vector<std::pair<REAL, std::vector<std::size_t> > > seed;
    std::vector<std::size_t> tmp;
    seed.push_back(std::make_pair(0,tmp));

    std::vector<std::pair<REAL, std::vector<std::size_t> > > ret;

    // for each n-gram length
    for(std::size_t i = 0; i < scores.size(); ++i){

        std::vector<std::pair<REAL, std::vector<std::size_t> > > scores_current;

        //for each word in vocab (already pruned in prepare_hypotheses)
        for(std::size_t j = 0; j < scores[i].size(); ++j){

            //for each hypothesis we kept from (n-gram-length-1)
            for(std::size_t k = 0; k < seed.size(); ++k){

                std::vector<size_t> tempvect (seed[k].second);
                tempvect.push_back(scores[i][j].second);

                scores_current.push_back(std::make_pair(seed[k].first + log(scores[i][j].first), tempvect));
            }
        }

        //we only need Nbest hypotheses
        std::nth_element(scores_current.begin(), min(scores_current.end(),scores_current.begin()+Nbest), scores_current.end(), std::greater<std::pair<REAL, std::vector<std::size_t> > >());
        seed.resize(std::min(Nbest,scores_current.size()));

        REAL length_penalty = weight_lengths(input_length,i+1);
        for(std::size_t j = 0; j < std::min(Nbest,scores_current.size()); ++j) {
            ret.push_back(std::make_pair((scores_current[j].first+length_penalty)/(i+1), scores_current[j].second)); // normalized by length
            seed[j] = scores_current[j]; // unnormalized; used to generate longer hypotheses
        }

    }

    // compare n-grams of different lengths and return Nbest
    std::sort(ret.begin(), ret.end(), std::greater<std::pair<REAL, std::vector<std::size_t> > >());
    ret.resize(std::min(ret.size(),Nbest));

    return ret;
}

