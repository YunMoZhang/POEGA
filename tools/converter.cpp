#include <algorithm>    // std::shuffle
#include "../include/commons/globals.hpp"


bool IsWeightedFormat(string format)
{
	if((format == "bwcsr")	||
		(format == "wcsr")	||
		(format == "wel"))
			return true;
	return false;
}

string GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

int main(int argc, char** argv)
{
	if(argc!= 4)
	{
		cout << "\nThere was an error parsing command line arguments\n";
		exit(0);
	}
	
	string input = string(argv[1]);
	string output = string(argv[2]);
	int64_t skip_n = atoll(argv[3]);;
	
	if(GetFileExtension(input) == "el")
	{
		ifstream infile;
		infile.open(input);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		
		vector<std::pair<uint, uint>> edges;
		uint u_, v_;
		string s1;
		while(skip_n>0){
			getline(infile, s1);
			cout << s1 <<endl;
			skip_n--;
		}
		while(getline( infile, line ))
		{
			ss.str("");
			ss.clear();
			ss << line;
			
			ss >> u_;
			ss >> v_;
			// ss >> newEdge.w8;
			// if(ss >> u_ >> u_){
				if(u_ <= v_)
					edges.push_back(std::make_pair(u_, v_));
				else
					edges.push_back(std::make_pair(v_, u_));
				edgeCounter++;

				if (max < u_)
					max = u_;

				if (max < v_)
					max = v_;	
			// }
			// else{
			// 	cout << "error: " << u_ << "\t" << v_ <<  "\t" << line << "\t line:" << edgeCounter << endl;
			// }
		}			
		infile.close();
		uint num_nodes = max + 1;
		cout << "Read #lines: " << edgeCounter << "; #node: " << num_nodes << endl;

		// 1)sorting
		std::sort(edges.begin(), edges.end());
		cout << "Finish sorting, begin deduplication" << endl;
		// 2)dedup
		uint num_edges = 0;
		uint last_u = edges[0].first, last_v = edges[0].second;
		for(uint i=1; i < edgeCounter; i++) {
			if(edges[i].first == last_u && edges[i].second == last_v) continue;
			// removing self loops
			// if(edge_list[i].first == edge_list[i].second) continue;
			// u_edge_list[num_edges++] = edge_list[i];
			edges[num_edges++] = edges[i];
			last_u = edges[i].first;
			last_v = edges[i].second;
		}
		cout << "Finish dedup, output lines: " << num_edges << endl;
		cout << "begin shuffling" << endl;
		// 3)shuffle
		unsigned seed = 111;
		std::shuffle(edges.begin(), edges.begin() + (num_edges-1), std::default_random_engine(seed)); 

		std::ofstream outfile(output+".bin", std::ofstream::binary);
		
		outfile.write((char*)&num_nodes, sizeof(uint));
		outfile.write((char*)&num_edges, sizeof(uint));
		for(uint i=0; i<num_edges; i++){
			outfile.write((char*)&edges[i].first, sizeof(uint));
			outfile.write((char*)&edges[i].second, sizeof(uint));
		}
		outfile.close();
		cout << "[Done]. Written to " << output << ".bin" << endl;
	}
	else
	{
		cout << "\nInput file format is not supported.\n";
		exit(0);
	}

}
