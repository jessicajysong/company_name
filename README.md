# company_name

### Required Packages
#### Please make sure the following packages are installed. 

- pandas
- string
- cleanco
- collections
- fuzzywuzzy
- matplotlib

If needed, please install with a following command:<br />
    '''pip install (package name to install)'''<br /><br />
     i.e To install the 'cleanco' package, run the following:<br />
    <code> 'pip install cleanco' <code/>


### Assumptions & Methods
#### Assumptions
(1) Different companies can use the same word in the name
(2) Companies do not have the accented and special characters in their names
(3) Names with match score greater than 80 are regarded as a match
(4) Difference in an order of the words within a name is not penalized

#### Methods
- Used 'Levenshtein' distance to calculate the match score
