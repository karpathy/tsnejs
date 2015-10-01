
# tSNEJS

tSNEJS is an implementation of t-SNE visualization algorithm in Javascript. 

t-SNE is a visualization algorithm that embeds things in 2 or 3 dimensions. If you have some data and you can measure their pairwise differences, t-SNE visualization can help you identify clusters in your data. See example below.

## Online demo
The main project website has a [live example](http://cs.stanford.edu/people/karpathy/tsnejs/) and more description.

There is also the [t-SNE CSV demo](http://cs.stanford.edu/people/karpathy/tsnejs/csvdemo.html) that allows you to simply paste CSV data into a textbox and tSNEJS computes and visualizes the embedding on the fly (no coding needed).

## Research Paper
The algorithm was originally described in this paper:

    L.J.P. van der Maaten and G.E. Hinton.
    Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research
    9(Nov):2579-2605, 2008.

You can find the PDF [here](http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

## Example code
Import tsne.js into your document: `<script src="tsne.js"></script>`
And then here is some example code:

```javascript

var opt = {}
opt.epsilon = 10; // epsilon is learning rate (10 = default)
opt.perplexity = 30; // roughly how many neighbors each point influences (30 = default)
opt.dim = 2; // dimensionality of the embedding (2 = default)

var tsne = new tsnejs.tSNE(opt); // create a tSNE instance

// initialize data. Here we have 3 points and some example pairwise dissimilarities
var dists = [[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.1, 1.0]];
tsne.initDataDist(dists);

for(var k = 0; k < 500; k++) {
  tsne.step(); // every time you call this, solution gets better
}

var Y = tsne.getSolution(); // Y is an array of 2-D points that you can plot
```

The data can be passed to tSNEJS as a set of high-dimensional points using the `tsne.initDataRaw(X)` function, where X is an array of arrays (high-dimensional points that need to be embedded). The algorithm computes the Gaussian kernel over these points and then finds the appropriate embedding.

## Web Demos
There are two web interfaces to this library that we are aware of:

- By Andrej, [here](http://cs.stanford.edu/people/karpathy/tsnejs/csvdemo.html).
- By Laurens, [here](http://homepage.tudelft.nl/19j49/tsnejs/), which takes data in different format and can also use Google Spreadsheet input.

## About
Send questions to [@karpathy](https://twitter.com/karpathy).

## License

MIT

