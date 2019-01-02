/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.graph.models.deepwalk;

import java.beans.ConstructorProperties;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.deeplearning4j.graph.api.IGraph;
import org.deeplearning4j.graph.api.IVertexSequence;
import org.deeplearning4j.graph.api.NoEdgeHandling;
import org.deeplearning4j.graph.api.Vertex;
import org.deeplearning4j.graph.iterator.GraphWalkIterator;
import org.deeplearning4j.graph.iterator.parallel.GraphWalkIteratorProvider;
import org.deeplearning4j.graph.iterator.parallel.RandomWalkGraphIteratorProvider;
import org.deeplearning4j.graph.models.embeddings.GraphVectorLookupTable;
import org.deeplearning4j.graph.models.embeddings.GraphVectorsImpl;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DeepWalk<V, E> extends GraphVectorsImpl<V, E> {
    public static final int STATUS_UPDATE_FREQUENCY = 1000;
    private Logger log = LoggerFactory.getLogger(DeepWalk.class);
    private int vectorSize;
    private int windowSize;
    private double learningRate;
    private boolean initCalled = false;
    private long seed;
    private ExecutorService executorService;
    private int nThreads = Runtime.getRuntime().availableProcessors();
    private transient AtomicLong walkCounter = new AtomicLong(0L);

    public DeepWalk() {
    }

    public int getVectorSize() {
        return this.vectorSize;
    }

    public int getWindowSize() {
        return this.windowSize;
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        if (this.lookupTable != null) {
            this.lookupTable.setLearningRate(learningRate);
        }

    }

    public void initialize(IGraph<V, E> graph) {
        int nVertices = graph.numVertices();
        int[] degrees = new int[nVertices];

        for(int i = 0; i < nVertices; ++i) {
            degrees[i] = graph.getVertexDegree(i);
        }

        this.initialize(degrees);
    }

    public void initialize(int[] graphVertexDegrees) {
        this.log.info("Initializing: Creating Huffman tree and lookup table...");
        GraphHuffman gh = new GraphHuffman(graphVertexDegrees.length);
        gh.buildTree(graphVertexDegrees);
        this.lookupTable = new InMemoryGraphLookupTable(graphVertexDegrees.length, this.vectorSize, gh, this.learningRate);
        this.initCalled = true;
        this.log.info("Initialization complete");
    }

    public void fit(IGraph<V, E> graph, int walkLength) {
        if (!this.initCalled) {
            this.initialize(graph);
        }

        GraphWalkIteratorProvider<V> iteratorProvider = new RandomWalkGraphIteratorProvider(graph, walkLength, this.seed, NoEdgeHandling.SELF_LOOP_ON_DISCONNECTED);
        this.fit((GraphWalkIteratorProvider)iteratorProvider);
    }

    public void fit(GraphWalkIteratorProvider<V> iteratorProvider) {
        if (!this.initCalled) {
            throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        } else {
            List<GraphWalkIterator<V>> iteratorList = iteratorProvider.getGraphWalkIterators(this.nThreads);
            this.executorService = Executors.newFixedThreadPool(this.nThreads, new ThreadFactory() {
                public Thread newThread(Runnable r) {
                    Thread t = new Thread(r);
                    t.setDaemon(true);
                    return t;
                }
            });
            List<Future<Void>> list = new ArrayList(iteratorList.size());
            Iterator var4 = iteratorList.iterator();

            while(var4.hasNext()) {
                GraphWalkIterator<V> iter = (GraphWalkIterator)var4.next();
                LearningCallable c = new LearningCallable(iter);
                list.add(this.executorService.submit(c));
            }

            this.executorService.shutdown();

            try {
                this.executorService.awaitTermination(999L, TimeUnit.DAYS);
            } catch (InterruptedException var8) {
                throw new RuntimeException("ExecutorService interrupted", var8);
            }

            var4 = list.iterator();

            while(var4.hasNext()) {
                Future f = (Future)var4.next();

                try {
                    f.get();
                } catch (Exception var7) {
                    throw new RuntimeException(var7);
                }
            }

        }
    }

    public void fit(GraphWalkIterator<V> iterator) {
        if (!this.initCalled) {
            throw new UnsupportedOperationException("DeepWalk not initialized (call initialize before fit)");
        } else {
            int walkLength = iterator.walkLength();

            while(iterator.hasNext()) {
                IVertexSequence<V> sequence = iterator.next();
                int[] walk = new int[walkLength + 1];

                for(int var5 = 0; sequence.hasNext(); walk[var5++] = ((Vertex)sequence.next()).vertexID()) {
                }

                this.skipGram(walk);
                long iter = this.walkCounter.incrementAndGet();
                if (iter % 1000L == 0L) {
                    this.log.info("Processed {} random walks on graph", iter);
                }
            }

        }
    }

    private void skipGram(int[] walk) {
        for(int mid = this.windowSize; mid < walk.length - this.windowSize; ++mid) {
            for(int pos = mid - this.windowSize; pos <= mid + this.windowSize; ++pos) {
                if (pos != mid) {
                    this.lookupTable.iterate(walk[mid], walk[pos]);
                }
            }
        }

    }

    public GraphVectorLookupTable lookupTable() {
        return this.lookupTable;
    }

    private class LearningCallable implements Callable<Void> {
        private final GraphWalkIterator<V> iterator;

        public Void call() throws Exception {
            DeepWalk.this.fit(this.iterator);
            return null;
        }

        @ConstructorProperties({"iterator"})
        public LearningCallable(GraphWalkIterator<V> iterator) {
            this.iterator = iterator;
        }
    }

    public static class Builder<V, E> {
        private int vectorSize = 100;
        private long seed = System.currentTimeMillis();
        private double learningRate = 0.01D;
        private int windowSize = 2;

        public Builder() {
        }

        public Builder<V, E> vectorSize(int vectorSize) {
            this.vectorSize = vectorSize;
            return this;
        }

        public Builder<V, E> learningRate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }

        public Builder<V, E> windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder<V, E> seed(long seed) {
            this.seed = seed;
            return this;
        }

        public DeepWalk<V, E> build() {
            DeepWalk<V, E> dw = new DeepWalk();
            dw.vectorSize = this.vectorSize;
            dw.windowSize = this.windowSize;
            dw.learningRate = this.learningRate;
            dw.seed = this.seed;
            return dw;
        }
    }
}
