LLMs are good at making the graphs for standard architectures - they are good at variable tracking, and especially good with translating syntaxes. 

But they break down with "**dynamic control flow**" - if and while loops lead to hallucinations.
For **massive architectures**, with hundreds of nested modules and intricate skip connections (DenseNets or complex Multimodal architectures) - not enough **context** window.
