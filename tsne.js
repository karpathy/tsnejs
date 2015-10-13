// create main global object
var tsnejs = tsnejs || { REVISION: 'ALPHA' };

(function(global) {
  "use strict";

  // utility function
  var assert = function(condition, message) {
    if (!condition) { throw message || "Assertion failed"; }
  }

  // syntax sugar
  var getopt = function(opt, field, defaultval) {
    if(opt.hasOwnProperty(field)) {
      return opt[field];
    } else {
      return defaultval;
    }
  }

  // return 0 mean unit standard deviation random number
  var return_v = false;
  var v_val = 0.0;
  var gaussRandom = function() {
    if(return_v) { 
      return_v = false;
      return v_val; 
    }
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if(r == 0 || r > 1) return gaussRandom();
    var c = Math.sqrt(-2*Math.log(r)/r);
    v_val = v*c; // cache this for next function call for efficiency
    return_v = true;
    return u*c;
  }

  // return random normal number
  var randn = function(mu, std){ return mu+gaussRandom()*std; }

  // utilitity that creates contiguous vector of zeros of size n
  var zeros = function(n) {
    if(typeof(n)==='undefined' || isNaN(n)) { return []; }
    if(typeof ArrayBuffer === 'undefined') {
      // lacking browser support
      var arr = new Array(n);
      for(var i=0;i<n;i++) { arr[i]= 0; }
      return arr;
    } else {
      return new Float64Array(n); // typed arrays are faster
    }
  }

  // utility that returns 2d array filled with random numbers
  // or with value s, if provided
  var randn2d = function(n,d,s) {
    var uses = typeof s !== 'undefined';
    var x = [];
    for(var i=0;i<n;i++) {
      var xhere = [];
      for(var j=0;j<d;j++) { 
        if(uses) {
          xhere.push(s); 
        } else {
          xhere.push(randn(0.0, 1e-4)); 
        }
      }
      x.push(xhere);
    }
    return x;
  }

  // compute L2 distance between two vectors
  var L2 = function(x1, x2) {
    var D = x1.length;
    var d = 0;
    for(var i=0;i<D;i++) { 
      var x1i = x1[i];
      var x2i = x2[i];
      d += (x1i-x2i)*(x1i-x2i);
    }
    return d;
  }

  // compute pairwise distance in all vectors in X
  var xtod = function(X) {
    var N = X.length;
    var dist = zeros(N * N); // allocate contiguous array
    for(var i=0;i<N;i++) {
      for(var j=i+1;j<N;j++) {
        var d = L2(X[i], X[j]);
        dist[i*N+j] = d;
        dist[j*N+i] = d;
      }
    }
    return dist;
  }

  // compute (p_{i|j} + p_{j|i})/(2n)
  var d2p = function(D, perplexity, tol) {
    var Nf = Math.sqrt(D.length); // this better be an integer
    var N = Math.floor(Nf);
    assert(N === Nf, "D should have square number of elements.");
    var Htarget = Math.log(perplexity); // target entropy of distribution
    var P = zeros(N * N); // temporary probability matrix

    var prow = zeros(N); // a temporary storage compartment
    for(var i=0;i<N;i++) {
      var betamin = -Infinity;
      var betamax = Infinity;
      var beta = 1; // initial value of precision
      var done = false;
      var maxtries = 50;

      // perform binary search to find a suitable precision beta
      // so that the entropy of the distribution is appropriate
      var num = 0;
      while(!done) {
        //debugger;

        // compute entropy and kernel row with beta precision
        var psum = 0.0;
        for(var j=0;j<N;j++) {
          var pj = Math.exp(- D[i*N+j] * beta);
          if(i===j) { pj = 0; } // we dont care about diagonals
          prow[j] = pj;
          psum += pj;
        }
        // normalize p and compute entropy
        var Hhere = 0.0;
        for(var j=0;j<N;j++) {
          var pj = prow[j] / psum;
          prow[j] = pj;
          if(pj > 1e-7) Hhere -= pj * Math.log(pj);
        }

        // adjust beta based on result
        if(Hhere > Htarget) {
          // entropy was too high (distribution too diffuse)
          // so we need to increase the precision for more peaky distribution
          betamin = beta; // move up the bounds
          if(betamax === Infinity) { beta = beta * 2; }
          else { beta = (beta + betamax) / 2; }

        } else {
          // converse case. make distrubtion less peaky
          betamax = beta;
          if(betamin === -Infinity) { beta = beta / 2; }
          else { beta = (beta + betamin) / 2; }
        }

        // stopping conditions: too many tries or got a good precision
        num++;
        if(Math.abs(Hhere - Htarget) < tol) { done = true; }
        if(num >= maxtries) { done = true; }
      }

      // console.log('data point ' + i + ' gets precision ' + beta + ' after ' + num + ' binary search steps.');
      // copy over the final prow to P at row i
      for(var j=0;j<N;j++) { P[i*N+j] = prow[j]; }

    } // end loop over examples i

    // symmetrize P and normalize it to sum to 1 over all ij
    var Pout = zeros(N * N);
    var N2 = N*2;
    for(var i=0;i<N;i++) {
      for(var j=0;j<N;j++) {
        Pout[i*N+j] = Math.max((P[i*N+j] + P[j*N+i])/N2, 1e-100);
      }
    }

    return Pout;
  }

  // helper function
  function sign(x) { return x > 0 ? 1 : x < 0 ? -1 : 0; }

  var tSNE = function(opt) {
    var opt = opt || {};
    this.perplexity = getopt(opt, "perplexity", 30); // effective number of nearest neighbors
    this.dim = getopt(opt, "dim", 2); // by default 2-D tSNE
    this.epsilon = getopt(opt, "epsilon", 10); // learning rate

    this.iter = 0;
  }

  tSNE.prototype = {

    // this function takes a set of high-dimensional points
    // and creates matrix P from them using gaussian kernel
    initDataRaw: function(X) {
      var N = X.length;
      var D = X[0].length;
      assert(N > 0, " X is empty? You must have some data!");
      assert(D > 0, " X[0] is empty? Where is the data?");
      var dists = xtod(X); // convert X to distances using gaussian kernel
      this.P = d2p(dists, this.perplexity, 1e-4); // attach to object
      this.N = N; // back up the size of the dataset
      this.initSolution(); // refresh this
    },

    // this function takes a given distance matrix and creates
    // matrix P from them.
    // D is assumed to be provided as a list of lists, and should be symmetric
    initDataDist: function(D) {
      var N = D.length;
      assert(N > 0, " X is empty? You must have some data!");
      // convert D to a (fast) typed array version
      var dists = zeros(N * N); // allocate contiguous array
      for(var i=0;i<N;i++) {
        for(var j=i+1;j<N;j++) {
          var d = D[i][j];
          dists[i*N+j] = d;
          dists[j*N+i] = d;
        }
      }
      this.P = d2p(dists, this.perplexity, 1e-4);
      this.N = N;
      this.initSolution(); // refresh this
    },

    // (re)initializes the solution to random
    initSolution: function() {
      // generate random solution to t-SNE
      this.Y = randn2d(this.N, this.dim); // the solution
      this.gains = randn2d(this.N, this.dim, 1.0); // step gains to accelerate progress in unchanging directions
      this.ystep = randn2d(this.N, this.dim, 0.0); // momentum accumulator
      this.iter = 0;
    },

    // return pointer to current solution
    getSolution: function() {
      return this.Y;
    },

    // perform a single step of optimization to improve the embedding
    step: function() {
      this.iter += 1;
      var N = this.N;

      var cg = this.costGrad(this.Y); // evaluate gradient
      var cost = cg.cost;
      var grad = cg.grad;

      // perform gradient step
      var ymean = zeros(this.dim);
      for(var i=0;i<N;i++) {
        for(var d=0;d<this.dim;d++) {
          var gid = grad[i][d];
          var sid = this.ystep[i][d];
          var gainid = this.gains[i][d];

          // compute gain update
          var newgain = sign(gid) === sign(sid) ? gainid * 0.8 : gainid + 0.2;
          if(newgain < 0.01) newgain = 0.01; // clamp
          this.gains[i][d] = newgain; // store for next turn

          // compute momentum step direction
          var momval = this.iter < 250 ? 0.5 : 0.8;
          var newsid = momval * sid - this.epsilon * newgain * grad[i][d];
          this.ystep[i][d] = newsid; // remember the step we took

          // step!
          this.Y[i][d] += newsid; 

          ymean[d] += this.Y[i][d]; // accumulate mean so that we can center later
        }
      }

      // reproject Y to be zero mean
      for(var i=0;i<N;i++) {
        for(var d=0;d<this.dim;d++) {
          this.Y[i][d] -= ymean[d]/N;
        }
      }

      //if(this.iter%100===0) console.log('iter ' + this.iter + ', cost: ' + cost);
      return cost; // return current cost
    },

    // for debugging: gradient check
    debugGrad: function() {
      var N = this.N;

      var cg = this.costGrad(this.Y); // evaluate gradient
      var cost = cg.cost;
      var grad = cg.grad;

      var e = 1e-5;
      for(var i=0;i<N;i++) {
        for(var d=0;d<this.dim;d++) {
          var yold = this.Y[i][d];

          this.Y[i][d] = yold + e;
          var cg0 = this.costGrad(this.Y);

          this.Y[i][d] = yold - e;
          var cg1 = this.costGrad(this.Y);
          
          var analytic = grad[i][d];
          var numerical = (cg0.cost - cg1.cost) / ( 2 * e );
          console.log(i + ',' + d + ': gradcheck analytic: ' + analytic + ' vs. numerical: ' + numerical);

          this.Y[i][d] = yold;
        }
      }
    },

    // return cost and gradient, given an arrangement
    costGrad: function(Y) {
      var N = this.N;
      var dim = this.dim; // dim of output space
      var P = this.P;

      var pmul = this.iter < 100 ? 4 : 1; // trick that helps with local optima

      // compute current Q distribution, unnormalized first
      var Qu = zeros(N * N);
      var qsum = 0.0;
      for(var i=0;i<N;i++) {
        for(var j=i+1;j<N;j++) {
          var dsum = 0.0;
          for(var d=0;d<dim;d++) {
            var dhere = Y[i][d] - Y[j][d];
            dsum += dhere * dhere;
          }
          var qu = 1.0 / (1.0 + dsum); // Student t-distribution
          Qu[i*N+j] = qu;
          Qu[j*N+i] = qu;
          qsum += 2 * qu;
        }
      }
      // normalize Q distribution to sum to 1
      var NN = N*N;
      var Q = zeros(NN);
      for(var q=0;q<NN;q++) { Q[q] = Math.max(Qu[q] / qsum, 1e-100); }

      var cost = 0.0;
      var grad = [];
      for(var i=0;i<N;i++) {
        var gsum = new Array(dim); // init grad for point i
        for(var d=0;d<dim;d++) { gsum[d] = 0.0; }
        for(var j=0;j<N;j++) {
          cost += - P[i*N+j] * Math.log(Q[i*N+j]); // accumulate cost (the non-constant portion at least...)
          var premult = 4 * (pmul * P[i*N+j] - Q[i*N+j]) * Qu[i*N+j];
          for(var d=0;d<dim;d++) {
            gsum[d] += premult * (Y[i][d] - Y[j][d]);
          }
        }
        grad.push(gsum);
      }

      return {cost: cost, grad: grad};
    }
  }

  global.tSNE = tSNE; // export tSNE class
})(tsnejs);


// export the library to window, or to module in nodejs
(function(lib) {
  "use strict";
  if (typeof module === "undefined" || typeof module.exports === "undefined") {
    window.tsnejs = lib; // in ordinary browser attach library to window
  } else {
    module.exports = lib; // in nodejs
  }
})(tsnejs);
