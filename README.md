# Tree LSTM : A Long Short Memory architecture for Tree structure data 

# Acknowledgments:
    - The original Tree-LSTM architecture was presented in this paper : https://arxiv.org/abs/1503.00075

    -  The full source code for the original work presented here : https://github.com/stanfordnlp/treelstm  

    - Python implementation was presented here https://github.com/dasguptar/treelstm.pytorch

# Why a new Python implementation  ?

    - The python implementation is quite messed up ! Various functionalities are not separated out 
      and in summary need refactoring which is exactly what is done here.

    - I strongly believe that if you can re-factor a code, you can understand it !

# Main features : 
   - Various functionalities are clearly separated out. You can easily modify 'stanford_parser.py'
     to parse any text file if you have Stanford CoreNLP components ready.

   - In case you do not have the stanford CoreNLP infrastructure you can use the already parsed files
     to try TreeLSTM. 
 
   - The code is written a 'pythonic' way and all the parameters are passed using a 'config' file
     in place of hardcoded in the code itself.


# How to run ?

   - If you have Stanford CoreNLP components you can use edit 'get_prams' of stanford_parser.py and then 
     run  
   > `python stanford_parser.py`   

   - For running the main code use:

  > `python driver.py -c config.ini' 


# Comments & Feedback 

   - You can write to me (prasad.jayanti@gmail.com) for any feedback. 


LocalWords:  LSTM
