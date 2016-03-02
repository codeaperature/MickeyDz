// Import trig functions
/*

Find the clusters of McDonalds Restaurants.

Given this data:

head mcdonalds.csv
-149.868307,61.140133,"McDonalds [WM]-Anchorage,AK","8900 Old Seward Hwy [WM], Anchorage,AK, (907) 344-5831"
-149.88113,61.192426,"McDonalds [WM]-Anchorage,AK","3101 A St [WM], Anchorage,AK, (907) 561-5137"
-149.898633,61.194753,"McDonalds-Anchorage,AK","800 W Northern Lights Blvd, Anchorage,AK, (907) 561-5525"
-149.870565,61.188348,"McDonalds-Anchorage,AK","701 E 36th Ave, Anchorage,AK, (907) 563-0658"

Convert Lat / Long values into X,Y,Z coordinates
Use KMeans clustering to determine the cluster centers.
Note: In a previous exercise, the optimal number of clusters was found to be about 120.
This number was determined by checking the error. The error amount always goes down as 
more clusters are formed. However, after a certain point, the error doesn't reduce as fast.
The elbow point is what I am considering as the optimal point.

So I will use the value of 120 for the number of clusters. The umber of iterations was 

For each store calculate its distance to the centers 
and label it with the center it is closest to.


Find the center that has the most stores.
Find the store that is closest to this center.


FINDINGS:

The cluster that has the most stores can change at random ... 

Considering that cluster centers formations are based randomly-select-and-then-
iterate process, the this means there are no hard answers as to what value will be a 
cluster's center. This means that a cluster can have slightly more stores in a cluster
on a random basis.

Subsequent runs of the following code yield varying results:


The McDonalds that is closest to the biggest cluster of 120 clusters is:
 "McDonalds-New York,NY","208 Varick St, New York,NY, (212) 206-9991"
 
The McDonalds that is closest to the biggest cluster of 120 clusters is:
 "McDonalds-River Edge,NJ","1118 Main St, River Edge,NJ, (201) 489-9216"
 
The McDonalds that is closest to the biggest cluster of 120 clusters is:
 "McDonalds-The Villages,FL","320 Colony Blvd, The Villages,FL, (352) 753-9225"

The McDonalds that is closest to the biggest cluster of 120 clusters is:
 "McDonalds-Englewood,NJ","41 W Palisade Ave, Englewood,NJ, (201) 894-0048"

The McDonalds that is closest to the biggest cluster of 120 clusters is:
 "McDonalds-Tarrytown,NY","140 Wildey St, Tarrytown,NY, (914) 631-9620"
 
 

*/

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}    
import Math.{PI,cos,sin,sqrt}    
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import com.github.fommil.netlib.BLAS;

case class CoordCC(x:Double, y:Double, z:Double)
case class IndivMDzCC(info:String, coord:CoordCC)

val radius = 3959.0
val radianPerDegree = PI / 180.0

val pathMD = "./mcdonalds.csv"
val dataMD = sc.textFile(pathMD)
val numClusters = 120 // empirically found
val numIterations = 30

val cnt = dataMD.count
val eachAllDataMD = dataMD.
  map(str => {
    val latEnd = str.indexOf(",")
    val lonEnd = str.indexOf(",", latEnd+1)
    val lat = radianPerDegree * str.substring(0, latEnd).toDouble
    val lon = radianPerDegree * str.substring(latEnd + 1, lonEnd).toDouble
    val tem = radius * cos(lat) 
    val x = tem * cos(lon)
    val y = tem * sin(lon)
    val z = radius * sin(lat)
    IndivMDzCC(str.substring(lonEnd+1), CoordCC(x,y,z))
  }).cache

val eachCoordMD = eachAllDataMD.map(each => Vectors.dense(Array(each.coord.x,each.coord.y,each.coord.z))).cache

val model = KMeans.train(eachCoordMD, numClusters, numIterations)

val centers = Array.ofDim[CoordCC](numClusters)
var i = 0
for(i <- 0 until numClusters) {
  centers(i) = CoordCC(model.clusterCenters(i)(0), model.clusterCenters(i)(1), model.clusterCenters(i)(2))
}

val centersRDD = sc.parallelize(Array(centers))
val cart = eachAllDataMD.cartesian(centersRDD)

/*

scala> cart.map(x => x._1.info).collect
res98: Array[String] = Array("McDonalds [WM]-Anchorage,AK","8900 Old Seward Hwy [WM], Anchorage,AK, (907) 344-5831", "McDonalds [WM]-Anchorage,AK","3101 A St [WM], Anchorage,AK, (907) 561-5137", "McDonalds-Anchorage,AK","800 W Northern Lights Blvd, Anchorage,AK, (907) 561-5525", "McDonalds-Anchorage,AK","701 E 36th Ave, Anchorage,AK, (907) 563-0658", "McDonalds-Anchorage,AK","8915 Old Seward Hwy, Anchorage,AK, (907) 344-4231", "McDonalds-Anchorage,AK","1320 Huffman Rd, Anchorage,AK, (907) 345-5932", "McDonalds-Anchorage,AK","2601 E Tudor Rd, Anchorage,AK, (907) 562-2108", "McDonalds-Anchorage,AK","3006 Mountain View Dr, Anchorage,AK, (907) 274-5686", "McDonalds-Anchorage,AK","5716 Debarr Rd, Anchorage,AK, (907) 333-0413", "McDonalds-Anchorage,AK","255 Muldoon Rd, Anchorage,AK, (907) 337...
scala> 

cart.map(x => (x._1.info, x._2(0))).collect
res100: Array[(String, CoordCC)] = Array(("McDonalds [WM]-Anchorage,AK","8900 Old Seward Hwy [WM], Anchorage,AK, (907) 344-5831",CoordCC(60.70071076847053,48.11477758734534,-3958.212439647893)), ("McDonalds [WM]-Anchorage,AK","3101 A St [WM], Anchorage,AK, (907) 561-5137",CoordCC(60.70071076847053,48.11477758734534,-3958.212439647893)), ("McDonalds-Anchorage,AK","800 W Northern Lights Blvd, Anchorage,AK, (907) 561-5525",CoordCC(60.70071076847053,48.11477758734534,-3958.212439647893)), ("McDonalds-Anchorage,AK","701 E 36th Ave, Anchorage,AK, (907) 563-0658",CoordCC(60.70071076847053,48.11477758734534,-3958.212439647893)), ("McDonalds-Anchorage,AK","8915 Old Seward Hwy, Anchorage,AK, (907) 344-4231",CoordCC(60.70071076847053,48.11477758734534,-3958.212439647893)), ("McDonalds-Anchorage,AK...
scala> 

*/


val rel2Center = cart.map(cartItem => {
  var i = 0;
  var minDist = radius
  var minCenter = 0
  for(i <- 0 until cartItem._2.length) {
    val dx = cartItem._1.coord.x - cartItem._2(i).x
    val dy = cartItem._1.coord.y - cartItem._2(i).y
    val dz = cartItem._1.coord.z - cartItem._2(i).z
    val dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if(dist < minDist) {
       minDist = dist
       minCenter = i
    }
  }
  (cartItem._1, minCenter, minDist)
})

/*
scala> rel2Center.map(store => (store._2, 1)).reduceByKey((a,b) => a + b).collect
res114: Array[(Int, Int)] = Array((96,92), (112,103), (16,447), (80,105), (48,164), (32,46), (0,223), (64,22), (81,102), (97,12), (65,74), (33,113), (113,37), (1,58), (17,478), (49,188), (34,11), (82,90), (66,137), (98,28), (50,4), (18,198), (114,8), (2,196), (19,313), (115,122), (35,105), (51,168), (83,59), (67,75), (3,218), (99,393), (84,101), (100,336), (52,138), (4,21), (116,18), (36,47), (20,34), (68,240), (101,4), (21,98), (53,90), (117,131), (37,228), (69,157), (85,131), (5,29), (22,95), (54,122), (102,9), (118,9), (6,273), (70,39), (38,112), (86,26), (39,4), (119,54), (71,148), (55,86), (23,285), (7,144), (103,21), (87,13), (56,138), (104,109), (24,167), (40,138), (72,56), (8,215), (88,35), (41,290), (25,35), (105,88), (73,28), (57,281), (89,212), (9,149), (42,77), (106,15), (74...
scala> 


The McDonalds that is closest to the biggest cluster of 100 clusters is:
 "McDonalds PlayPlace-Homer Glen,IL","14298 S Bell Rd, Homer Glen,IL, (708) 301-2332"

*/
val maxCenter = rel2Center.
  map(store  => (store._2, 1)).
  reduceByKey((a,b) => a + b).
  map(center => center.swap).
  sortByKey(false).
  take(1)


printf("Center #%d has %d stores\n", maxCenter(0)._2, maxCenter(0)._1)
val centerMost = rel2Center.
  filter(store => store._2 == maxCenter(0)._2).
  map(store => (store._3, store._1)).
  sortByKey(true).
  take(1)
  
printf("The McDonalds that is closest to the biggest cluster of %d clusters is:\n %s\n",
numClusters, centerMost(0)._2.info )
  