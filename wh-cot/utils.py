from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
import numpy as np
import re
import random
import pandas as pd
from collections import Counter
import unidecode
from typing import List, Tuple, Dict

punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

def remove_punc(text):
    exclude = set(punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    s = unidecode.unidecode(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))

#计算EM、f1分数
def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_evaluate(gold_answers: List[str], pred_answers: List[str]):
    em_scores = [0.0]
    f1_scores = [0.0]

    for pred_answer in pred_answers:
        for gold_answer in gold_answers:
            em_scores.append(compute_em(pred_answer, gold_answer))
            f1_scores.append(compute_f1(pred_answer, gold_answer))

    return max(em_scores), max(f1_scores)


def extract_answer(raw_pred: str):
    raw_pred = raw_pred.strip(' ')
    if raw_pred == "":
        return ""

    # find yes or no
    yes_no_answer = check_answer(raw_pred)
    if yes_no_answer is not None:
        return yes_no_answer

    # remove the words in ()
    raw_pred = re.sub("\([^)]*\)", "", raw_pred)
    #answers = re.split(", | and | or |/| either ", raw_pred)

    return raw_pred


def check_answer(text):
    pattern = r"^(yes|no)[,.]?"
    match = re.match(pattern, text, re.IGNORECASE)
    if match:
        answer = match.group(0).lower()  # Get the matched string in lowercase
        answer = re.sub(r'[,.]', '', answer)  # Remove punctuation
        return answer
    else:
        return None
    
def create_demo_text(task, cot_flag):
    x, z, y = [], [], []
    direct_answer_trigger_for_fewshot = "The answer is"
    # example sentences ...    
    if task in ("multiarith", "gsm8k"):
        
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.")
        y.append("8")

    elif task in ("svamp"):
        
        x.append("Paco had 26 salty cookies and 17 sweet cookies. He ate 14 sweet cookies and 9 salty cookies.How many salty cookies did Paco have left?")
        z.append("( 26.0 - 9.0 )")
        y.append("17.0")

        x.append("Zachary did 44 push-ups in gym class today. David did 58 more push-ups than zachary.How many push-ups did Zachary and David do altogether?")
        z.append("( ( 44.0 + 44.0 ) + 58.0 )")
        y.append("92.0")        

        x.append("Frank was reading through his favorite book. The book had 41 chapters, each with the same number of pages. It has a total of 450 pages. It took Frank 30 days to finish the book.How many pages did he read per day?")
        z.append("( 450.0 / 30.0 )")
        y.append("15.0")        

        x.append("The Razorback t-shirt shop sells each t-shirt for $ 51 dollars. During the Arkansas and Texas tech game they offered a discount of $ 8 per t-shirt and sold 130 t-shirts. How much money did they make from selling the t-shirts?")
        z.append("( ( 51.0 - 8.0 ) * 130.0 )")
        y.append("5590.0")   


    elif task in ("aqua"):
        
        x.append("there are more than 501 students in a school such that 20% of them exactly took physics and 28% of them exactly took math. What could be the least possible no of students in the school?\n Answer Choices: (A) 550 (B) 570 (C) 600 (D) 700 (E) none of these")
        z.append("20% means 1/5 and 28% means 7/25,taking the lcm of the denominators 5 and 25 we get 25,the least multiple of 25 which is greater than 501 is 525. So, answer is none\nANSWER:E")
        y.append("E")

        x.append("Two cars are travelling from the same starting point in the same direction, having started their commute at the same time. The first car travels at a steady rate of 55 mph, while the second travels at a steady rate of 52 mph. How much time will pass before the cars are 15 miles away from each other?\n Answer Choices: (A) 3 hours (B) 5 hours (C) 6 hours (D) 4 hours (E) 7 hours")
        z.append("Relative Speed: 55-52=3 mph\nDistance:15 miles\nTime: distance/speed=15/3= 5 hours\nCorrect answer is B")
        y.append("B")        

        x.append("Jerry purchased a 1-year $5,000 bond that paid an annual interest rate of 12% compounded every six months. How much interest had this bond accrued at maturity?\n Answer Choices: (A) $5102 (B) $618 (C) $216 (D) $202 (E) $200")
        z.append("A=P(1+r/q)nq .Here q is no of times interest is compounded in a yr so it is = 2. Amount comes out to be 5618 .Hence interest is 5618-5000=618. >>B")
        y.append("B")        

        x.append("A paper is in a square form whose one side is 20 cm. Two semi circles are drawn on its opposites as diameters. If these semi circles are cut down what is the area of the remaining paper?\n Answer Choices: (A) 8.75 (B) 8.79 (C) 8.75 (D) 8.71 (E) 8.72")
        z.append("(5 * 3.5)/2 = 8.75\nAnswer:C")
        y.append("C")   

    elif task in ("hotpot"):
        
        x.append("Minor league baseball games that were played between the rivalling teams of the Sooners and the Cowboys were played at the stadium formerly known as what?")
        z.append("The Bedlam Series refers to the athletics rivalry between the University of Oklahoma Sooners and the Oklahoma State University Cowboys of the Big 12 Conference. For a number of years Drillers Stadium also hosted one of the regular season baseball games played between Oklahoma State University and the University of Oklahoma in the Bedlam Series.")
        y.append("Drillers Stadium")

        x.append("Among the cast for \"Suicide Squad\", who has also appeared in \"Flags of Our Fathers\"?")
        z.append("Scott Eastwood | He has appeared in the films \"Flags of Our Fathers\" (2006), \"Gran Torino\" (2008), \"Invictus\" (2009), \"The Forger\" (2012), \"Trouble with the Curve\" (2012), \"Texas Chainsaw\" (2013), \"Fury\" (2014), \"The Perfect Wave\" (2014), \"The Longest Ride\" (2015), \"Mercury Plains\" (2016), \"Suicide Squad\" (2016), \"Snowden\" (2016), \"Walk of Fame\" (2017), and \"The Fate of the Furious\" (2017). Suicide Squad (film) | The film is written and directed by David Ayer and stars an ensemble cast featuring Will Smith, Jared Leto, Margot Robbie, Joel Kinnaman, Viola Davis, Jai Courtney, Jay Hernandez, Adewale Akinnuoye-Agbaje, Ike Barinholtz, Scott Eastwood, and Cara Delevingne.")
        y.append("Scott Eastwood")        

        x.append("Do Lush and P.O.D. both consist of four band members?")
        z.append("Payable on Death (abbreviated P.O.D.) is a Christian nu metal band formed in 1992 and based in San Diego, California. Lush were an English rock band formed in London in 1987. The lineup before the original split consisted of Miki Berenyi (vocals, guitar), Emma Anderson (vocals, guitar), Phil King (bass) and Chris Acland (drums). The band's line-up consists of vocalist Sonny Sandoval, drummer Wuv Bernardo, guitarist Marcos Curiel, and bassist Traa Daniels.")
        y.append("yes")        

        x.append("Who was the great grandfather of Franklin Seaver Pratt's wife?")
        z.append("He married Elizabeth Keka\u02bbaniau La\u02bbanui, a member of the Hawaiian nobility, and defended her claims to the Hawaiian crown lands during the overthrow of the Kingdom of Hawaii. Elizabeth Keka\u02bbaniau La\u02bbanui Pratt, full name Elizabeth Kekaikuihala Keka\u02bbaniauokalani Kalaninuiohilaukapu La\u02bbanui Pratt (12 September 1834 \u2013 20 December 1928) was a great grandniece of Kamehameha I, being a great granddaughter of Kalokuokamaile, the older brother of Kamehameha I, founder of the Kingdom of Hawaii.")
        y.append("Kalokuokamaile")    
    
    elif task in ("musique"):
        
        x.append("who has had the most hits in the league that includes the team that has played the most games of the type played just before the league MVP is awarded?")
        z.append("after the World Series they give out the mlb mvp award.  the New York Yankees played in the most #1 games. Major League Baseball is the league of #2. Pete Rose has the most hits in #3 history.")
        y.append("Pete Rose")

        x.append("In what county is the city that shares a border with the capital of the state where Levi Casey was born?")
        z.append("Levi Casey >> place of birth is South Carolina. Columbia became the state capital of #1. #2 >> shares border with  Forest Acres. Richland County located in the administrative territorial entity.")
        y.append("Richland County")        

        x.append("Can foreign tourists cruise to the country between the nation that hosted 2020 AFC U-23 Championship qualification and the one that contains A Don.?")
        z.append("Thailand hosted the tournament. A Don is a village in south-eastern Laos near the border with Vietnam. It is located in Kaleum District in Sekong Province. Myanmar (also known as Burma) is the northwestern-most country of mainland Southeast Asia, bordering China, India, Bangladesh, Thailand and Laos.")
        y.append("It is not possible for foreigners to go to/from Myanmar by sea or river.")        

        x.append("Who sings Home Alone Tonight with the singer of Light It Up?")
        z.append("``Light It Up ''is a song by American country music artist Luke Bryan. It is the lead single to his sixth studio album, What Makes You Country. Bryan wrote the song with Brad Tursi of the band Old Dominion.``Home Alone Tonight ''is a song recorded by American country music artist Luke Bryan as a duet with Karen Fairchild of American country music group Little Big Town for his fifth studio album, Kill the Lights (2015). Upon the release of the album, the song entered the Billboard Hot Country Songs chart at number 33 on the strength of digital downloads. It was serviced to American country radio on November 23, 2015 as the album's third official single.")
        y.append("Karen Fairchild")    
    
    elif task in ("2wikim"):
        
        x.append("Who died first, Gustav Philipp M\u00f6rl or Giuseppe Diotti?")
        z.append("Gustav Philipp M\u00f6rl->date of death->May 7, 1750. Giuseppe Diotti->date of death->30 January 1846")
        y.append("Gustav Philipp M\u00f6rl")

        x.append("Are the directors of both films Captain Calamity (Film) and Heaven Only Knows (Film) from the same country?")
        z.append("Captain Calamity (film)->director->John Reinhardt. Heaven Only Knows (film)->director->Albert S. Rogell. John Reinhardt (director)->country of citizenship->American. Albert S. Rogell->country of citizenship->American.")
        y.append("yes")        

        x.append("Where did the composer of film Rajaratha graduate from?")
        z.append("Rajaratha->composer->Anup Bhandari. Anup Bhandari->educated at->Vidya Vikas Institute of Engineering & Technology (VVIET).")
        y.append("Vidya Vikas Institute of Engineering & Technology (VVIET)")        

        x.append("What is the date of death of the director of film Det Kunne V\u00e6rt Deg?")
        z.append("Det kunne v\u00e6rt deg->director->Henki Kolstad. Henki Kolstad->date of death->14 July 2008")
        y.append("14 July 2008")    
    

    else:
        raise ValueError("dataset is not properly defined ...")
        
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += "Question: " + x[i] + "\nAnswer: " + z[i] + " " + \
                         direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Question: " + x[i] + "\nAnswer: " + \
                         direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text
