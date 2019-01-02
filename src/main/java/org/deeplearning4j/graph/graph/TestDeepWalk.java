package org.deeplearning4j.graph.graph;

import org.deeplearning4j.graph.data.GraphLoader;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.RandomWalkIterator;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Arrays;


public class TestDeepWalk {

    public void basicTest(){

        int vectorSize = 5;
        int windowSize = 2;
        int walkLength = 8;
        double learningRate = 0.01;
        int vertices = 10312;

       // ClassPathResource cpr = new ClassPathResource("/home/cogbot-developer/singnet/DeepWalkTest/src/dataset/BlogCatalog-dataset/data/edges.csv");
        String fileEdges= "/home/cogbot-developer/singnet/DeepWalkTest/src/dataset/BlogCatalog-dataset/data/edges.csv";

        Graph<String,String> graph = null;
        try {
            graph = GraphLoader.loadUndirectedGraphEdgeListFile(fileEdges, vertices, ",");
        } catch (IOException e) {
            e.printStackTrace();
        }

        DeepWalk<String,String> deepWalk = new DeepWalk.Builder()
                .vectorSize(vectorSize)
                .windowSize(windowSize)
                .learningRate(learningRate)
                .build();
        deepWalk.initialize(graph);

        for( int i=0; i<vertices; i++ ){
            INDArray vector = deepWalk.getVertexVector(i);

            System.out.println(Arrays.toString(vector.dup().data().asFloat()));
        }

        GraphWalkIterator<String> iter = new RandomWalkIterator<>(graph,walkLength);

        deepWalk.fit(iter);

        for( int t=0; t<5; t++ ) {
            iter.reset();
            deepWalk.fit(iter);
            System.out.println("--------------------");
            for (int i = 0; i < vertices; i++) {
                INDArray vector = deepWalk.getVertexVector(i);

                System.out.println(Arrays.toString(vector.dup().data().asFloat()));
            }
        }
    }

    public static void main(String[] args) {

        TestDeepWalk dw = new TestDeepWalk();
        dw.basicTest();
    }

}
