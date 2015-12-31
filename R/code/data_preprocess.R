setwd(dirname(parent.frame(2)$ofile))
library(dplyr)
library(ggplot2)
library(xgboost)
library(data.table)
library(e1071)

xgbDMTrain=xgb.DMatrix(as.matrix(train[,xFeatures]),label=train$logExpected,missing=NA)

SparseM::as.matrix(c$x)

trainF <- "../../data_CSV/train.csv"
testF <- "../../data_CSV/test.csv"
require(bit64)
train <- rbindlist(lapply(trainF, fread, sep=","))
test <- rbindlist(lapply(testF, fread, sep=","))
train <- train[-647055]
test <- test[-653647]

vaibhavF <- "../../data_CSV/vaibhav.csv"
ismailF <- "../../data_CSV/ismail.csv"
yeF <- "../../data_CSV/ye.csv"
vaibhav <- rbindlist(lapply(vaibhavF, fread, sep=","))
ismail <- rbindlist(lapply(ismailF, fread, sep=","))
ye <- rbindlist(lapply(yeF, fread, sep=","))

Id <- 1:length(vaibhav$Expected)
predictions <- data.frame(Id)
predictions$Expected <- ismail$Expected*0.5 + ye$Expected*0.5
dtTime <- gsub(":", "-", Sys.time())
write.csv(predictions,paste("sample_solution_",dtTime,".csv"),row.names=F)



# temp code to plot the adjacency matrix
library(tcltk)
library(cccd)
dist <- read.csv('dist.csv',header = FALSE)
setZeroMin <- function(col) {
  col_sort <- sort(col)
  min <- 0.9999
  if (col_sort[3]<min) {
    min <- col_sort[3]
  }
  col[col>min] <- 0
  col[col==1] <- 0
  return(col)
}
dist_new <- sapply(dist,setZeroMin)


cols <- c("FINANCIAL SERVICES", "SHOES", "PERSONAL CARE", "PAINT AND ACCESSORIES", "DSD GROCERY", "MEAT - FRESH & FROZEN", "DAIRY", "PETS AND SUPPLIES", "HOUSEHOLD CHEMICALS/SUPP", "NULL", "IMPULSE MERCHANDISE", "PRODUCE", "CANDY, TOBACCO, COOKIES", "GROCERY DRY GOODS", "BOYS WEAR", "FABRICS AND CRAFTS", "JEWELRY AND SUNGLASSES", "MENS WEAR", "ACCESSORIES", "HOME MANAGEMENT", "FROZEN FOODS", "SERVICE DELI", "INFANT CONSUMABLE HARDLINES", "PRE PACKED DELI", "COOK AND DINE", "PHARMACY OTC", "LADIESWEAR", "COMM BREAD", "BAKERY", "HOUSEHOLD PAPER GOODS", "CELEBRATION", "HARDWARE", "BEAUTY", "AUTOMOTIVE", "BOOKS AND MAGAZINES", "SEAFOOD", "OFFICE SUPPLIES", "LAWN AND GARDEN", "SHEER HOSIERY", "WIRELESS", "BEDDING", "BATH AND SHOWER", "HORTICULTURE AND ACCESS", "HOME DECOR", "TOYS", "INFANT APPAREL", "LADIES SOCKS", "PLUS AND MATERNITY", "ELECTRONICS", "GIRLS WEAR, 4-6X  AND 7-14", "BRAS & SHAPEWEAR", "LIQUOR,WINE,BEER", "SLEEPWEAR/FOUNDATIONS", "CAMERAS AND SUPPLIES", "SPORTING GOODS", "PLAYERS AND ELECTRONICS", "PHARMACY RX", "MENSWEAR", "OPTICAL - FRAMES", "SWIMWEAR/OUTERWEAR", "OTHER DEPARTMENTS", "MEDIA AND GAMING", "FURNITURE", "OPTICAL - LENSES", "SEASONAL", "LARGE HOUSEHOLD GOODS", "1-HR PHOTO", "CONCEPT STORES", "HEALTH AND BEAUTY AIDS")
colnames(dist_new) <- cols

id <- tkplot(graph_from_adjacency_matrix(as.matrix(dist_new),mode = 'undirected',diag = FALSE, weighted = TRUE)
       ,canvas.width=1200, canvas.height=650)


id <- tkplot(g, layout=lay)
canvas <- tk_canvas(id)
tkpostscript(canvas, file="/tmp/output.eps")
