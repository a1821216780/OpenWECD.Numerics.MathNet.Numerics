// <copyright file="Matrix.cs" company="Math.NET">
// Math.NET Numerics, part of the Math.NET Project
// http://numerics.mathdotnet.com
// http://github.com/mathnet/mathnet-numerics
//
// Copyright (c) 2009-2015 Math.NET
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// </copyright>

using System;
using MathNet.Numerics.LinearAlgebra.Double.Factorization;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace MathNet.Numerics.LinearAlgebra.Int
{
    /// <summary>
    /// <c>double</c> version of the <see cref="Matrix{T}"/> class.
    /// </summary>
    [Serializable]
    public abstract class Matrix : Matrix<int>
    {
        /// <summary>
        /// Initializes a new instance of the Matrix class.
        /// </summary>
        protected Matrix(MatrixStorage<int> storage)
            : base(storage)
        {
        }

        /// <summary>
        /// Set all values whose absolute value is smaller than the threshold to zero.
        /// </summary>
        public override void CoerceZero(double threshold)
        {
            MapInplace(x => Math.Abs(x) < threshold ? 0 : x, Zeros.AllowSkip);
        }

        /// <summary>
        /// Returns the conjugate transpose of this matrix.
        /// </summary>
        /// <returns>The conjugate transpose of this matrix.</returns>
        public sealed override Matrix<int> ConjugateTranspose()
        {
            return Transpose();
        }

        /// <summary>
        /// Puts the conjugate transpose of this matrix into the result matrix.
        /// </summary>
        public sealed override void ConjugateTranspose(Matrix<int> result)
        {
            Transpose(result);
        }

        /// <summary>
        /// Complex conjugates each element of this matrix and place the results into the result matrix.
        /// </summary>
        /// <param name="result">The result of the conjugation.</param>
        protected sealed override void DoConjugate(Matrix<int> result)
        {
            if (ReferenceEquals(this, result))
            {
                return;
            }

            CopyTo(result);
        }

        /// <summary>
        /// Negate each element of this matrix and place the results into the result matrix.
        /// </summary>
        /// <param name="result">The result of the negation.</param>
        protected override void DoNegate(Matrix<int> result)
        {
            Map(x => -x, result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Add a scalar to each element of the matrix and stores the result in the result vector.
        /// </summary>
        /// <param name="scalar">The scalar to add.</param>
        /// <param name="result">The matrix to store the result of the addition.</param>
        protected override void DoAdd(int scalar, Matrix<int> result)
        {
            Map(x => x + scalar, result, Zeros.Include);
        }

        /// <summary>
        /// Adds another matrix to this matrix.
        /// </summary>
        /// <param name="other">The matrix to add to this matrix.</param>
        /// <param name="result">The matrix to store the result of the addition.</param>
        /// <exception cref="ArgumentNullException">If the other matrix is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">If the two matrices don't have the same dimensions.</exception>
        protected override void DoAdd(Matrix<int> other, Matrix<int> result)
        {
            Map2((x, y) => x + y, other, result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Subtracts a scalar from each element of the vector and stores the result in the result vector.
        /// </summary>
        /// <param name="scalar">The scalar to subtract.</param>
        /// <param name="result">The matrix to store the result of the subtraction.</param>
        protected override void DoSubtract(int scalar, Matrix<int> result)
        {
            Map(x => x - scalar, result, Zeros.Include);
        }

        /// <summary>
        /// Subtracts another matrix from this matrix.
        /// </summary>
        /// <param name="other">The matrix to subtract to this matrix.</param>
        /// <param name="result">The matrix to store the result of subtraction.</param>
        /// <exception cref="ArgumentNullException">If the other matrix is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">If the two matrices don't have the same dimensions.</exception>
        protected override void DoSubtract(Matrix<int> other, Matrix<int> result)
        {
            Map2((x, y) => x - y, other, result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Multiplies each element of the matrix by a scalar and places results into the result matrix.
        /// </summary>
        /// <param name="scalar">The scalar to multiply the matrix with.</param>
        /// <param name="result">The matrix to store the result of the multiplication.</param>
        protected override void DoMultiply(int scalar, Matrix<int> result)
        {
            Map(x => x*scalar, result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Multiplies this matrix with a vector and places the results into the result vector.
        /// </summary>
        /// <param name="rightSide">The vector to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected override void DoMultiply(Vector<int> rightSide, Vector<int> result)
        {
            for (var i = 0; i < RowCount; i++)
            {
                var s = 0;
                for (var j = 0; j < ColumnCount; j++)
                {
                    s += At(i, j)*rightSide[j];
                }
                result[i] = s;
            }
        }

        /// <summary>
        /// Divides each element of the matrix by a scalar and places results into the result matrix.
        /// </summary>
        /// <param name="divisor">The scalar to divide the matrix with.</param>
        /// <param name="result">The matrix to store the result of the division.</param>
        protected override void DoDivide(int divisor, Matrix<int> result)
        {
            Map(x => x/divisor, result, divisor == 0 ? Zeros.Include : Zeros.AllowSkip);
        }

        /// <summary>
        /// Divides a scalar by each element of the matrix and stores the result in the result matrix.
        /// </summary>
        /// <param name="dividend">The scalar to divide by each element of the matrix.</param>
        /// <param name="result">The matrix to store the result of the division.</param>
        protected override void DoDivideByThis(int dividend, Matrix<int> result)
        {
            Map(x => dividend/x, result, Zeros.Include);
        }

        /// <summary>
        /// Multiplies this matrix with another matrix and places the results into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected override void DoMultiply(Matrix<int> other, Matrix<int> result)
        {
            for (var i = 0; i < RowCount; i++)
            {
                for (var j = 0; j < other.ColumnCount; j++)
                {
                    var s = 0;
                    for (var k = 0; k < ColumnCount; k++)
                    {
                        s += At(i, k)*other.At(k, j);
                    }
                    result.At(i, j, s);
                }
            }
        }

        /// <summary>
        /// Multiplies this matrix with transpose of another matrix and places the results into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected override void DoTransposeAndMultiply(Matrix<int> other, Matrix<int> result)
        {
            for (var j = 0; j < other.RowCount; j++)
            {
                for (var i = 0; i < RowCount; i++)
                {
                    var s = 0;
                    for (var k = 0; k < ColumnCount; k++)
                    {
                        s += At(i, k)*other.At(j, k);
                    }
                    result.At(i, j, s);
                }
            }
        }

        /// <summary>
        /// Multiplies this matrix with the conjugate transpose of another matrix and places the results into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected sealed override void DoConjugateTransposeAndMultiply(Matrix<int> other, Matrix<int> result)
        {
            DoTransposeAndMultiply(other, result);
        }

        /// <summary>
        /// Multiplies the transpose of this matrix with another matrix and places the results into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected override void DoTransposeThisAndMultiply(Matrix<int> other, Matrix<int> result)
        {
            for (var j = 0; j < other.ColumnCount; j++)
            {
                for (var i = 0; i < ColumnCount; i++)
                {
                    var s = 0;
                    for (var k = 0; k < RowCount; k++)
                    {
                        s += At(k, i)*other.At(k, j);
                    }
                    result.At(i, j, s);
                }
            }
        }

        /// <summary>
        /// Multiplies the transpose of this matrix with another matrix and places the results into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected sealed override void DoConjugateTransposeThisAndMultiply(Matrix<int> other, Matrix<int> result)
        {
            DoTransposeThisAndMultiply(other, result);
        }

        /// <summary>
        /// Multiplies the transpose of this matrix with a vector and places the results into the result vector.
        /// </summary>
        /// <param name="rightSide">The vector to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected override void DoTransposeThisAndMultiply(Vector<int> rightSide, Vector<int> result)
        {
            for (var j = 0; j < ColumnCount; j++)
            {
                var s = 0;
                for (var i = 0; i < RowCount; i++)
                {
                    s += At(i, j)*rightSide[i];
                }
                result[j] = s;
            }
        }

        /// <summary>
        /// Multiplies the conjugate transpose of this matrix with a vector and places the results into the result vector.
        /// </summary>
        /// <param name="rightSide">The vector to multiply with.</param>
        /// <param name="result">The result of the multiplication.</param>
        protected sealed override void DoConjugateTransposeThisAndMultiply(Vector<int> rightSide, Vector<int> result)
        {
            DoTransposeThisAndMultiply(rightSide, result);
        }

        /// <summary>
        /// Computes the canonical modulus, where the result has the sign of the divisor,
        /// for the given divisor each element of the matrix.
        /// </summary>
        /// <param name="divisor">The scalar denominator to use.</param>
        /// <param name="result">Matrix to store the results in.</param>
        protected override void DoModulus(int divisor, Matrix<int> result)
        {
            Map(x => Euclid.Modulus(x, divisor), result, Zeros.Include);
        }

        /// <summary>
        /// Computes the canonical modulus, where the result has the sign of the divisor,
        /// for the given dividend for each element of the matrix.
        /// </summary>
        /// <param name="dividend">The scalar numerator to use.</param>
        /// <param name="result">A vector to store the results in.</param>
        protected override void DoModulusByThis(int dividend, Matrix<int> result)
        {
            Map(x => Euclid.Modulus(dividend, x), result, Zeros.Include);
        }

        /// <summary>
        /// Computes the remainder (% operator), where the result has the sign of the dividend,
        /// for the given divisor each element of the matrix.
        /// </summary>
        /// <param name="divisor">The scalar denominator to use.</param>
        /// <param name="result">Matrix to store the results in.</param>
        protected override void DoRemainder(int divisor, Matrix<int> result)
        {
            Map(x => Euclid.Remainder(x, divisor), result, Zeros.Include);
        }

        /// <summary>
        /// Computes the remainder (% operator), where the result has the sign of the dividend,
        /// for the given dividend for each element of the matrix.
        /// </summary>
        /// <param name="dividend">The scalar numerator to use.</param>
        /// <param name="result">A vector to store the results in.</param>
        protected override void DoRemainderByThis(int dividend, Matrix<int> result)
        {
            Map(x => Euclid.Remainder(dividend, x), result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise multiplies this matrix with another matrix and stores the result into the result matrix.
        /// </summary>
        /// <param name="other">The matrix to pointwise multiply with this one.</param>
        /// <param name="result">The matrix to store the result of the pointwise multiplication.</param>
        protected override void DoPointwiseMultiply(Matrix<int> other, Matrix<int> result)
        {
            Map2((x, y) => x*y, other, result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Pointwise divide this matrix by another matrix and stores the result into the result matrix.
        /// </summary>
        /// <param name="divisor">The matrix to pointwise divide this one by.</param>
        /// <param name="result">The matrix to store the result of the pointwise division.</param>
        protected override void DoPointwiseDivide(Matrix<int> divisor, Matrix<int> result)
        {
            Map2((x, y) => x/y, divisor, result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise raise this matrix to an exponent and store the result into the result matrix.
        /// </summary>
        /// <param name="exponent">The exponent to raise this matrix values to.</param>
        /// <param name="result">The matrix to store the result of the pointwise power.</param>
        protected override void DoPointwisePower(int exponent, Matrix<int> result)
        {
            Map(x => (int)Math.Pow((double)x, (double)exponent), result, exponent > 0.0 ? Zeros.AllowSkip : Zeros.Include);
        }

        /// <summary>
        /// Pointwise raise this matrix to an exponent and store the result into the result matrix.
        /// </summary>
        /// <param name="exponent">The exponent to raise this matrix values to.</param>
        /// <param name="result">The vector to store the result of the pointwise power.</param>
        protected override void DoPointwisePower(Matrix<int> exponent, Matrix<int> result)
        {
            Map2((x, y) => (int)Math.Pow((double)x, (double)y), result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise canonical modulus, where the result has the sign of the divisor,
        /// of this matrix with another matrix and stores the result into the result matrix.
        /// </summary>
        /// <param name="divisor">The pointwise denominator matrix to use</param>
        /// <param name="result">The result of the modulus.</param>
        protected override void DoPointwiseModulus(Matrix<int> divisor, Matrix<int> result)
        {
            Map2(Euclid.Modulus, divisor, result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise remainder (% operator), where the result has the sign of the dividend,
        /// of this matrix with another matrix and stores the result into the result matrix.
        /// </summary>
        /// <param name="divisor">The pointwise denominator matrix to use</param>
        /// <param name="result">The result of the modulus.</param>
        protected override void DoPointwiseRemainder(Matrix<int> divisor, Matrix<int> result)
        {
            Map2(Euclid.Remainder, divisor, result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise applies the exponential function to each value and stores the result into the result matrix.
        /// </summary>
        /// <param name="result">The matrix to store the result.</param>
        protected override void DoPointwiseExp(Matrix<int> result)
        {
            Map(x => (int)Math.Exp((double)x), result, Zeros.Include);
        }

        /// <summary>
        /// Pointwise applies the natural logarithm function to each value and stores the result into the result matrix.
        /// </summary>
        /// <param name="result">The matrix to store the result.</param>
        protected override void DoPointwiseLog(Matrix<int> result)
        {
            Map(x => (int)Math.Log((double)x), result, Zeros.Include);
        }

        protected override void DoPointwiseAbs(Matrix<int> result)
        {
            Map(Math.Abs, result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseAcos(Matrix<int> result)
        {
            Map(x => (int)Math.Acos((double)x), result, Zeros.Include);
        }
        protected override void DoPointwiseAsin(Matrix<int> result)
        {
            Map(x => (int)Math.Asin((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseAtan(Matrix<int> result)
        {
            Map(x => (int)Math.Atan((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseAtan2(Matrix<int> other, Matrix<int> result)
        {
            Map2((x,y) => (int)Math.Atan2((double)x, (double)y), other, result, Zeros.Include);
        }
        protected override void DoPointwiseCeiling(Matrix<int> result)
        {
            Map(x => (int)Math.Ceiling((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseCos(Matrix<int> result)
        {
            Map(x => (int)Math.Cos((double)x), result, Zeros.Include);
        }
        protected override void DoPointwiseCosh(Matrix<int> result)
        {
            Map(x => (int)Math.Cosh((double)x), result, Zeros.Include);
        }
        protected override void DoPointwiseFloor(Matrix<int> result)
        {
            Map(x => (int)Math.Floor((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseLog10(Matrix<int> result)
        {
            Map(x => (int)Math.Log10((double)x), result, Zeros.Include);
        }
        protected override void DoPointwiseRound(Matrix<int> result)
        {
            Map(x => (int)Math.Round((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseSign(Matrix<int> result)
        {
            Map(x => Math.Sign(x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseSin(Matrix<int> result)
        {
            Map(x => (int)Math.Sin((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseSinh(Matrix<int> result)
        {
            Map(x => (int)Math.Sinh((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseSqrt(Matrix<int> result)
        {
            Map(x => (int)Math.Sqrt((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseTan(Matrix<int> result)
        {
            Map(x => (int)Math.Tan((double)x), result, Zeros.AllowSkip);
        }
        protected override void DoPointwiseTanh(Matrix<int> result)
        {
            Map(x => (int)Math.Tanh((double)x), result, Zeros.AllowSkip);
        }

        /// <summary>
        /// Computes the Moore-Penrose Pseudo-Inverse of this matrix.
        /// </summary>
        public override Matrix<int> PseudoInverse()
        {
            var svd = Svd(true);
            var w = svd.W;
            var s = svd.S;
            double tolerance = Math.Max(RowCount, ColumnCount) * svd.L2Norm * Precision.DoublePrecision;

            for (int i = 0; i < s.Count; i++)
            {
                s[i] = s[i] < tolerance ? 0 : 1/s[i];
            }

            w.SetDiagonal(s);
            return (svd.U * (w * svd.VT)).Transpose();
        }

        /// <summary>
        /// Computes the trace of this matrix.
        /// </summary>
        /// <returns>The trace of this matrix</returns>
        /// <exception cref="ArgumentException">If the matrix is not square</exception>
        public override int Trace()
        {
            if (RowCount != ColumnCount)
            {
                throw new ArgumentException("Matrix must be square.");
            }

            var sum = 0;
            for (var i = 0; i < RowCount; i++)
            {
                sum += At(i, i);
            }

            return sum;
        }

        protected override void DoPointwiseMinimum(int scalar, Matrix<int> result)
        {
            Map(x => Math.Min(scalar, x), result, scalar >= 0d ? Zeros.AllowSkip : Zeros.Include);
        }

        protected override void DoPointwiseMaximum(int scalar, Matrix<int> result)
        {
            Map(x => Math.Max(scalar, x), result, scalar <= 0d ? Zeros.AllowSkip : Zeros.Include);
        }

        protected override void DoPointwiseAbsoluteMinimum(int scalar, Matrix<int> result)
        {
            int absolute = Math.Abs(scalar);
            Map(x => Math.Min(absolute, Math.Abs(x)), result, Zeros.AllowSkip);
        }

        protected override void DoPointwiseAbsoluteMaximum(int scalar, Matrix<int> result)
        {
            int absolute = Math.Abs(scalar);
            Map(x => Math.Max(absolute, Math.Abs(x)), result, Zeros.Include);
        }

        protected override void DoPointwiseMinimum(Matrix<int> other, Matrix<int> result)
        {
            Map2(Math.Min, other, result, Zeros.AllowSkip);
        }

        protected override void DoPointwiseMaximum(Matrix<int> other, Matrix<int> result)
        {
            Map2(Math.Max, other, result, Zeros.AllowSkip);
        }

        protected override void DoPointwiseAbsoluteMinimum(Matrix<int> other, Matrix<int> result)
        {
            Map2((x, y) => Math.Min(Math.Abs(x), Math.Abs(y)), other, result, Zeros.AllowSkip);
        }

        protected override void DoPointwiseAbsoluteMaximum(Matrix<int> other, Matrix<int> result)
        {
            Map2((x, y) => Math.Max(Math.Abs(x), Math.Abs(y)), other, result, Zeros.AllowSkip);
        }

        /// <summary>Calculates the induced L1 norm of this matrix.</summary>
        /// <returns>The maximum absolute column sum of the matrix.</returns>
        public override double L1Norm()
        {
            var norm = 0d;
            for (var j = 0; j < ColumnCount; j++)
            {
                var s = 0d;
                for (var i = 0; i < RowCount; i++)
                {
                    s += Math.Abs(At(i, j));
                }
                norm = Math.Max(norm, s);
            }
            return norm;
        }

        /// <summary>Calculates the induced infinity norm of this matrix.</summary>
        /// <returns>The maximum absolute row sum of the matrix.</returns>
        public override double InfinityNorm()
        {
            var norm = 0d;
            for (var i = 0; i < RowCount; i++)
            {
                var s = 0d;
                for (var j = 0; j < ColumnCount; j++)
                {
                    s += Math.Abs(At(i, j));
                }
                norm = Math.Max(norm, s);
            }
            return norm;
        }

        /// <summary>Calculates the entry-wise Frobenius norm of this matrix.</summary>
        /// <returns>The square root of the sum of the squared values.</returns>
        public override double FrobeniusNorm()
        {
            var transpose = Transpose();
            var aat = this*transpose;
            var norm = 0d;
            for (var i = 0; i < RowCount; i++)
            {
                norm += aat.At(i, i);
            }
            return Math.Sqrt(norm);
        }

        /// <summary>
        /// Calculates the p-norms of all row vectors.
        /// Typical values for p are 1.0 (L1, Manhattan norm), 2.0 (L2, Euclidean norm) and positive infinity (infinity norm)
        /// </summary>
        public override Vector<double> RowNorms(double norm)
        {
            if (norm <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(norm), "Value must be positive.");
            }

            var ret = new double[RowCount];
            if (norm == 2.0)
            {
                Storage.FoldByRowUnchecked(ret, (s, x) => s + x*x, (x, _) => Math.Sqrt(x), ret, Zeros.AllowSkip);
            }
            else if (norm == 1.0)
            {
                Storage.FoldByRowUnchecked(ret, (s, x) => s + Math.Abs(x), (x, _) => x, ret, Zeros.AllowSkip);
            }
            else if (double.IsPositiveInfinity(norm))
            {
                Storage.FoldByRowUnchecked(ret, (s, x) => Math.Max(s, Math.Abs(x)), (x, _) => x, ret, Zeros.AllowSkip);
            }
            else
            {
                double invnorm = 1.0/norm;
                Storage.FoldByRowUnchecked(ret, (s, x) => s + Math.Pow(Math.Abs(x), norm), (x, _) => Math.Pow(x, invnorm), ret, Zeros.AllowSkip);
            }
            return Vector<double>.Build.Dense(ret);
        }

        /// <summary>
        /// Calculates the p-norms of all column vectors.
        /// Typical values for p are 1.0 (L1, Manhattan norm), 2.0 (L2, Euclidean norm) and positive infinity (infinity norm)
        /// </summary>
        public override Vector<double> ColumnNorms(double norm)
        {
            if (norm <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(norm), "Value must be positive.");
            }

            var ret = new double[ColumnCount];
            if (norm == 2.0)
            {
                Storage.FoldByColumnUnchecked(ret, (s, x) => s + x*x, (x, _) => Math.Sqrt(x), ret, Zeros.AllowSkip);
            }
            else if (norm == 1.0)
            {
                Storage.FoldByColumnUnchecked(ret, (s, x) => s + Math.Abs(x), (x, _) => x, ret, Zeros.AllowSkip);
            }
            else if (double.IsPositiveInfinity(norm))
            {
                Storage.FoldByColumnUnchecked(ret, (s, x) => Math.Max(s, Math.Abs(x)), (x, _) => x, ret, Zeros.AllowSkip);
            }
            else
            {
                double invnorm = 1.0/norm;
                Storage.FoldByColumnUnchecked(ret, (s, x) => s + Math.Pow(Math.Abs(x), norm), (x, _) => Math.Pow(x, invnorm), ret, Zeros.AllowSkip);
            }
            return Vector<double>.Build.Dense(ret);
        }

        /// <summary>
        /// Normalizes all row vectors to a unit p-norm.
        /// Typical values for p are 1.0 (L1, Manhattan norm), 2.0 (L2, Euclidean norm) and positive infinity (infinity norm)
        /// </summary>
        public sealed override Matrix<int> NormalizeRows(double norm)
        {
            throw new NotSupportedException();
            //var norminv = ((DenseVectorStorage<double>)RowNorms(norm).Storage).Data;
            //for (int i = 0; i < norminv.Length; i++)
            //{
            //    norminv[i] = norminv[i] == 0d ? 1d : 1d/norminv[i];
            //}

            //var result = Build.SameAs(this, RowCount, ColumnCount);
            //Storage.MapIndexedTo(result.Storage, (i, _, x) => norminv[i]*x, Zeros.AllowSkip, ExistingData.AssumeZeros);
            //return result;
        }

        /// <summary>
        /// Normalizes all column vectors to a unit p-norm.
        /// Typical values for p are 1.0 (L1, Manhattan norm), 2.0 (L2, Euclidean norm) and positive infinity (infinity norm)
        /// </summary>
        public sealed override Matrix<int> NormalizeColumns(double norm)
        {
            throw new NotSupportedException();
            //var norminv = ((DenseVectorStorage<int>)ColumnNorms(norm).Storage).Data;
            //for (int i = 0; i < norminv.Length; i++)
            //{
            //    norminv[i] = norminv[i] == 0d ? 1d : 1d/norminv[i];
            //}

            //var result = Build.SameAs(this, RowCount, ColumnCount);
            //Storage.MapIndexedTo(result.Storage, (_, j, x) => norminv[j]*x, Zeros.AllowSkip, ExistingData.AssumeZeros);
            //return result;
        }

        /// <summary>
        /// Calculates the value sum of each row vector.
        /// </summary>
        public override Vector<int> RowSums()
        {
            var ret = new int[RowCount];
            Storage.FoldByRowUnchecked(ret, (s, x) => s + x, (x, _) => x, ret, Zeros.AllowSkip);
            return Vector<int>.Build.Dense(ret);
        }

        /// <summary>
        /// Calculates the absolute value sum of each row vector.
        /// </summary>
        public override Vector<int> RowAbsoluteSums()
        {
            var ret = new int[RowCount];
            Storage.FoldByRowUnchecked(ret, (s, x) => s + Math.Abs(x), (x, _) => x, ret, Zeros.AllowSkip);
            return Vector<int>.Build.Dense(ret);
        }

        /// <summary>
        /// Calculates the value sum of each column vector.
        /// </summary>
        public override Vector<int> ColumnSums()
        {
            var ret = new int[ColumnCount];
            Storage.FoldByColumnUnchecked(ret, (s, x) => s + x, (x, _) => x, ret, Zeros.AllowSkip);
            return Vector<int>.Build.Dense(ret);
        }

        /// <summary>
        /// Calculates the absolute value sum of each column vector.
        /// </summary>
        public override Vector<int> ColumnAbsoluteSums()
        {
            var ret = new int[ColumnCount];
            Storage.FoldByColumnUnchecked(ret, (s, x) => s + Math.Abs(x), (x, _) => x, ret, Zeros.AllowSkip);
            return Vector<int>.Build.Dense(ret);
        }

        /// <summary>
        /// Evaluates whether this matrix is Hermitian (conjugate symmetric).
        /// </summary>
        public sealed override bool IsHermitian()
        {
            return IsSymmetric();
        }

        public override Cholesky<int> Cholesky()
        {
            throw new NotSupportedException();
            //return UserCholesky.Create(this);
        }

        public override LU<int> LU()
        {
            throw new NotSupportedException();
            //return UserLU.Create(this);
        }

        public override QR<int> QR(QRMethod method = QRMethod.Thin)
        {
            throw new NotSupportedException();
            //return UserQR.Create(this, method);
        }

        public override GramSchmidt<int> GramSchmidt()
        {
            throw new NotSupportedException();
            //return UserGramSchmidt.Create(this);
        }

        public override Svd<int> Svd(bool computeVectors = true)
        {
            throw new NotSupportedException();
            //return UserSvd.Create(this, computeVectors);
        }

        public override Evd<int> Evd(Symmetricity symmetricity = Symmetricity.Unknown)
        {
            throw new NotSupportedException();
            //return UserEvd.Create(this, symmetricity);
        }
    }
}
