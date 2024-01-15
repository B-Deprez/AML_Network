import networkit as nk
import pandas as pd

def assign_att(u, att, val):
    att[u] = val

def betweenness_nk(G_nk):
    psp_list = []
    G_nk.forNodes(lambda u: psp_list.append(u))
    
    print("Calculating betweenness...")
    betweenness_full = nk.centrality.Betweenness(G_nk, normalized=True)
    betweenness_full.run()
    print("Done")
    
    betweenness_att = G_nk.attachNodeAttribute("betweenness_full", float)
    G_nk.forNodes(lambda u: assign_att(u, betweenness_att, betweenness_full.score(u)))
    
    betweenness_list = []
    G_nk.forNodes(lambda u: betweenness_list.append(betweenness_att[u]))
    
    betweenness_df = pd.DataFrame({"PSP":psp_list, "Betweenness":betweenness_list})
    
    return(betweenness_df)
    
def closeness_nk(G_nk):
    psp_list = []
    G_nk.forNodes(lambda u: psp_list.append(u))
    
    print("Calculating closeness...")
    closeness_full = nk.centrality.Closeness(G_nk, True, True)
    closeness_full.run()
    print("Done")
    
    closeness_att = G_nk.attachNodeAttribute("closeness_full", float)
    G_nk.forNodes(lambda u: assign_att(u, closeness_att, closeness_full.score(u)))
    
    closeness_list = []
    G_nk.forNodes(lambda u: closeness_list.append(closeness_att[u]))
    
    closeness_df = pd.DataFrame({"PSP":psp_list, "Closeness":closeness_list})
    
    return(closeness_df)

def eigenvector_nk(G_nk):
    psp_list = []
    G_nk.forNodes(lambda u: psp_list.append(u))
    
    print("Calculating eigenvector cent. ...")
    eigen_full = nk.centrality.EigenvectorCentrality(G_nk)
    eigen_full.run()
    print("Done")
    
    eigen_att = G_nk.attachNodeAttribute("eigen_full", float)
    G_nk.forNodes(lambda u: assign_att(u, eigen_att, eigen_full.score(u)))
    
    eigen_list = []
    G_nk.forNodes(lambda u: eigen_list.append(eigen_att[u]))
    
    eigen_df = pd.DataFrame({"PSP":psp_list, "Eigenvector":eigen_list})
    
    return(eigen_df)