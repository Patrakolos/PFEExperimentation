import operator
from NaturePackage import Nature as nat
from NaturePackage import Genome as gn
import time
import re
import random
class Fabriquant:
    def __init__(self,GN,DM,bourrer,scrutin="Condorcet"):

        self.dm=DM
        self.listbuffer=[]
        exp = "(\d[H\d]+)/"
        self.attlen= GN.identity[0]
        self.recette=re.findall(exp,GN.identity)
        self.genome=gn.Genome()
        self.incarnation=[]
        self.listbuffer=[]
        exp2="H\d+"
        for stg in self.recette:
            self.attlen=stg[0]
            gene = re.findall(exp2, stg)
            self.condorcet(stg, gene)##
            ##if(scrutin=="Condorcet"):
            ##    self.condorcet(stg,gene)
            ##if(scrutin=="maximum_averrage"):
            ##    self.max_average(stg,gene)
        if(bourrer):
            self.bourrage2()
       # print(self.listbuffer)
        ch=""
        for m in self.recette:
            ch=ch+m+"/"
        self.genome.identity=ch
        self.genome.incarnation=self.incarnation

        self.genome.isvalide=1
        self.genome.resultat=sorted(self.listbuffer)



    def getgenome(self):
        return self.genome



    def bourrage2(self):
        Vincanration = []
        bourlist = []
        k = 0
        while (k < len(self.incarnation)):
            if (int(self.incarnation[k][2]) > 0):
                lalist=[]
                for l in self.incarnation[k][1]:
                    lalist.append(l)
                bourlist.extend(lalist)
                Vincanration.append(self.incarnation[k])
            k = k + 1
        self.incarnation=Vincanration
        self.listbuffer=bourlist
        ainter=list(self.dm.inter)
        aunion=list(self.dm.union)
        for i in self.listbuffer:
            if(i in ainter):
                ainter.remove(i)
            if(i in aunion):
                aunion.remove(i)

        k = random.randint(len(ainter),len(aunion))


        for j in range(k):
            fait=0
            while(fait==0):
                p=random.randint(0,nat.Nature.maxH-1)
               # print("getmegaheuristic")
                gjj = self.dm.getMegaHeuristique(["H" + str(p)], 1)
                #print("apres getmegaheuristicsss")
                hierlist3 = gjj[list(gjj.keys())[0]]
                hh=0
                while(hh<len(hierlist3)):
                 #   print("dehors")
                    #if(intersect(self.listbuffer,list(hierlist3[hh][0]))==[]):
                        #print("__",hierlist3[hh][1],"__k",k)
                    if(intersect(self.listbuffer,list(hierlist3[hh][0]))==[] and hierlist3[hh][1]>=0):
                  #      print("___________dedaans")
                        self.listbuffer = self.listbuffer + list(hierlist3[hh][0])
                        gene = "1H" + str(p)
                        self.recette.append(gene)
                        self.incarnation.append((gene, hierlist3[hh][0], hierlist3[hh][1]))
                        hh = len(hierlist3) + 1
                        fait=1
                    else:hh=hh+1
        #print("---selfincarnation-apres",self.incarnation)

    def max_average(self,stg,gene):
        nb_h=len(gene)
        for mesure in gene:
            dictt=self.dm.getMegaHeuristique([mesure], int(self.attlen))
            hierlist = dictt[list(dictt.keys())[0]]
            for p in hierlist:
                dict = {}
                j=tuple(p[0])
                if(p[1]<0):
                    dict[j]=-2*len(gene)-1
                else:
                    if(dict.keys()!=[]):
                        if(list(dict.keys()).__contains__(j)):
                            dict[j]=dict[j]+p[1]/nb_h
                        else:
                            dict[j]=p[1]/nb_h
                    else:
                        dict={}
                        dict[j] = p[1] / nb_h

        resultat=sorted(dict.items(),key=lambda t: t[1],reverse=True)
        if resultat != []:
            ii=0
            intersection=True
            while(intersection and ii<len(resultat)):
                if(intersect(self.listbuffer,list(resultat[ii][0]))==[]):
                    if (resultat[ii][1] > 0):
                        self.incarnation.append((stg, resultat[ii][0], 1))
                    else:
                        self.incarnation.append((stg, resultat[ii][0], -1))
                    for l in resultat[ii][0]:
                        self.listbuffer.append(l)
                    intersection=False
                ii=ii+1



    def condorcet(self,stg,gene):
        condidats = set()
        laval=0
        for mesure in gene:
            dictt = self.dm.getMegaHeuristique([mesure], int(self.attlen))
            hierlist = dictt[list(dictt.keys())[0]]
            latol = 0
            cpt1 = 0
            while (cpt1 < len(hierlist) and latol != nat.Nature.Tol):
                trock = set(self.listbuffer)
                if (intersect(trock, set(list(hierlist[cpt1][0]))) == []):
                    a = tuple(hierlist[cpt1][0])
                    condidats.add(a)
                    laval=hierlist[cpt1][1]
                    latol = latol + 1
                cpt1 = cpt1 + 1
        tournoit = []
        if(len(condidats)==1):
            if (laval > 0):
                valide = True
            else:
                valide = False

        for joueur in list(condidats):
            score = 0
            adversaires = list(condidats)
            adversaires.remove(joueur)
            if(adversaires!=[]):
                for adversaire in adversaires:
                    valide = False
                    for j in gene:
                        dicttt = self.dm.getMegaHeuristique([j], int(self.attlen))
                        hierlistx = dicttt[list(dicttt.keys())[0]]
                        kk = 0
                        while (kk < len(hierlistx)):
                            if (hierlistx[kk][0] == joueur):
                                if (hierlist[kk][1] > 0):
                                    valide = True
                                kk = len(hierlistx) + 1
                                score = score + 1
                            else:
                                if (hierlistx[kk][0] == adversaire):
                                    kk = len(hierlistx) + 1
                                    score = score - 1
                            kk = kk + 1

            tournoit.append((joueur, score, valide))
        tournoit = sorted(tournoit, key=operator.itemgetter(1), reverse=True)
        if (tournoit != []):
            if (tournoit[0][2] == True):
                self.incarnation.append((stg, tournoit[0][0], 1))
            else:
                self.incarnation.append((stg, tournoit[0][0], -1))
            for i in tournoit[0][0]:
                self.listbuffer.append(i)
    #        print(self.listbuffer)



def intersect(a, b):
    return list(set(a) & set(b))
